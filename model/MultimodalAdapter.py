import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from segment_anything import sam_model_registry, SamPredictor
from super_gradients.training import models
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from model.attention_processor import AttnProcessor, MultiModalAttnProcessor, CNAttnProcessor
from model.resampler.resampler import Resampler


class MultimodalAdapter:
    def __init__(self, args, num_tokens=0):
        self.args = args
        self.num_tokens = num_tokens
        self.vae = AutoencoderKL.from_pretrained(
            self.args.vae_model_path).to(dtype=torch.float16)
        self.controlnet = ControlNetModel.from_pretrained(
            self.args.controlnet_model_path, torch_dtype=torch.float16)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.args.image_encoder_path).to(self.args.device, dtype=torch.float16)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
            steps_offset=1, )
        self.basepipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.args.base_sd_model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler, vae=self.vae,
            feature_extractor=None, safety_checker=None)

        self.ckpt = self.args.multimodal_adapter_path
        self.clip_image_processor = CLIPImageProcessor()
        self.face_proj_model = self.init_proj_face()
        self.cloth_proj_model = self.init_proj_cloth()

        # Set up multi-adapter related configurations.
        self.set_multi_adapter()
        self.load_multi_adapter()

    def yolo_nas_pose_prediction(self, reference_image):
        yolo_nas = models.get(self.args.yolo_nas_model_type, pretrained_weights="coco_pose").cuda()
        model_predictions = yolo_nas.predict(reference_image, conf=0.5)
        pose_point = model_predictions.prediction.poses

        # Re-plot the pose image
        list_x, list_y, t = [], [], []
        for i in range(pose_point.shape[1]):
            x, y = int(pose_point[:, i, 0]), int(pose_point[:, i, 1])
            list_x.append(x)
            list_y.append(y)
            t.append([x, y])

        mid1_x, mid2_x = int((list_x[5] + list_x[6]) // 2), int((list_x[11] + list_x[12]) // 2)
        mid1_y, mid2_y = int((list_y[5] + list_y[6]) // 2), int((list_y[11] + list_y[12]) // 2)
        list_x.append(mid1_x), list_x.append(mid2_x)
        list_y.append(mid1_y), list_y.append(mid2_y)

        poses = plt.figure(figsize=(reference_image.shape[1] / 100, reference_image.shape[0] / 100), dpi=100)
        plt.style.use('dark_background'), plt.axis('off')
        plt.xticks([]), plt.yticks([])
        plt.xlim(0, reference_image.shape[1]), plt.ylim(reference_image.shape[0], 0)
        plt.scatter(list_x, list_y, color='red')
        plt.plot([list_x[4], list_x[2], list_x[0], list_x[1], list_x[3]],
                 [list_y[4], list_y[2], list_y[0], list_y[1], list_y[3]], color='blue')
        plt.plot([list_x[10], list_x[8], list_x[6], list_x[5], list_x[7], list_x[9]],
                 [list_y[10], list_y[8], list_y[6], list_y[5], list_y[7], list_y[9]], color='blue')
        plt.plot([list_x[16], list_x[14], list_x[12], list_x[11], list_x[13], list_x[15]],
                 [list_y[16], list_y[14], list_y[12], list_y[11], list_y[13], list_y[15]], color='blue')
        plt.plot([list_x[0], list_x[17], list_x[18]], [list_y[0], list_y[17], list_y[18]], color='blue')

        # transfer to Image class
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        pose = Image.open(buf)
        pose = pose.convert('RGB')
        pose = pose.resize((reference_image.shape[1], reference_image.shape[0]))
        pose.save('pose.jpg')

        return list_x, list_y, t, pose

    def SAM_segmentation(self, image, list_x, list_y, list_comb):
        def show_mask(mask, ax):
            color = np.array([0 / 255, 0 / 255, 0 / 255, 0.5])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam = sam_model_registry[self.args.sam_model_type](
            checkpoint=self.args.sam_checkpoint_path).to(device=self.args.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        # Perform SAM segmentation according to the box prompt and point prompt.
        x_offset = abs(list_x[1] - list_x[2])
        x1 = int(max(list_x[0] - 5 * x_offset, 0))
        x2 = int(min(list_x[0] + 5 * x_offset, image.shape[1]))
        y1 = int(max(list_y[0] - 5 * x_offset, 0))
        y2 = int(min(list_y[0] + 5 * x_offset, image.shape[0]))

        # Use the predicted points and boxes from YOLO-NAS-Pose to perform SAM segmentation.
        input_point = np.array(
            [list_comb[0], list_comb[1], list_comb[2], list_comb[3], list_comb[4],
             [list_x[0], list_y[0] - 2 * x_offset]])
        input_label = np.array([1, 1, 1, 1, 1, 1])

        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )

        masks = ~masks
        plt.figure(figsize=(image.shape[1] / 50, image.shape[0] / 50))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        segment = Image.open(buf)
        segment = segment.convert('RGB')
        segment = segment.resize((image.shape[1], image.shape[0]))

        # Crop the faceID image
        x1 = int(list_x[0] - 2 * x_offset)
        x2 = int(list_x[0] + 2 * x_offset)
        y1 = int(list_y[0] - 2.5 * x_offset)
        y2 = int(list_y[0] + 1.5 * x_offset)
        face = segment.crop((x1, y1, x2, y2))
        face = face.resize((512, 512))
        segment.save('seg.jpg')
        face.save('face.jpg')

        return segment, face

    def init_proj_face(self):
        """ Convert face or cloth image into prompt embeds """
        """ cross_attention_dim=768, clip_embeddings_dim=1280 """
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.basepipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.args.device, dtype=torch.float16)
        return image_proj_model

    def init_proj_cloth(self):
        """ Convert face and cloth image into prompt embeds """
        """ cross_attention_dim=768, clip_embeddings_dim=1280 """
        image_proj_model = Resampler(
            dim=self.basepipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.basepipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.args.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_face_embeds(self, face_image=None, clip_image_embeds=None):
        if isinstance(face_image, Image.Image):
            face_image = [face_image]
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.args.device, dtype=torch.float16)

        """  Face Tensor (1,257,1280), using the last_hidden_state in CLIPVisionModelOutput """
        """  clip_image_embeds (1,257,1280) -> image_prompt_embeds (1,257,768) """
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.face_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.face_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_cloth_embeds(self, cloth_image=None, clip_image_embeds=None):
        if isinstance(cloth_image, Image.Image):
            cloth_image = [cloth_image]
        clip_image = self.clip_image_processor(images=cloth_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.args.device, dtype=torch.float16)

        """  clip_image_embeds (1,257,1280) -> image_prompt_embeds (1,16,768) """
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.cloth_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.cloth_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_multi_adapter(self):
        # load basepipeline unet
        unet = self.basepipe.unet
        attn_procs = {}
        hidden_size = 0

        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            #
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = MultiModalAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.args.device, dtype=torch.float16)

        unet.set_attn_processor(attn_procs)

        if hasattr(self.basepipe, "controlnet"):
            if isinstance(self.basepipe.controlnet, MultiControlNetModel):
                for controlnet in self.basepipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.basepipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_multi_adapter(self):
        state_dict = torch.load(self.ckpt, map_location="cpu")
        self.face_proj_model.load_state_dict(state_dict["face_proj"])
        self.cloth_proj_model.load_state_dict(state_dict["cloth_proj"])
        attn_layers = torch.nn.ModuleList(self.basepipe.unet.attn_processors.values())
        attn_layers.load_state_dict(state_dict["multimodal_adapter"])

    # The scale of image prompt
    def set_scale(self, face_scale, cloth_scale):
        for attn_processor in self.basepipe.unet.attn_processors.values():
            if isinstance(attn_processor, MultiModalAttnProcessor):
                attn_processor.face_scale = face_scale

                if self.args.inject_cloth_prompt:
                    attn_processor.cloth_scale = cloth_scale
                else:
                    attn_processor.cloth_scale = 0

    '''
    Multimodal Prompt Adapter Pipeline
    
    The inference structure of this pipeline is shown in asserts/architecture.jpg, 
    where the input contains text prompts and a photo of a persion. 
    After the images are fed into the pipeline, YOLO-NAS-POSE is used to extract human pose landmarks, 
    and SAM is used to segment the mask that captures the model's face. After cropping and alignment, 
    the images of the model's face are sent to the Image Encoder module, 
    and the facial features of the person are transformed into Face Prompt Embeds by the Resampler module. 
    Similar to the method for image processing, text prompt is also transformed into Text Prompt Embeds through the Encoder module and enters the Cross-Attention module. 
    Finally, Face Prompt Embeds are concatenated with Text Prompt Embeds and sent to the frozen U-Net.
    '''

    def MultimodalPipe(self, ref_img=None, seed=None,
                       face_scale=0, cloth_scale=0,
                       guidance_scale=7.5, num_inference_steps=30,
                       text_prompt=None, negative_prompt=None, clip_image_embeds=None, **kwargs):

        """ 0 init prompt scale and basepipeline. """
        self.set_scale(face_scale, cloth_scale)
        self.basepipe.to(self.args.device)

        """ 1 Utilize YOLO-NAS-POSE to generate predicted points and draw the pose image. """
        x_points, y_points, combined_points, pose_img = self.yolo_nas_pose_prediction(ref_img)

        """ 2 Perform SAM segmentation and crop the facial image of a person in the picture. """
        _, face_img = self.SAM_segmentation(ref_img, x_points, y_points, combined_points)

        """ 3 Handle multimodal prompts, and transfer prompts into embeds. """
        if text_prompt is None:
            text_prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        face_prompt_embeds, uncond_face_prompt_embeds = self.get_face_embeds(
            face_image=face_img, clip_image_embeds=clip_image_embeds)
        if self.args.inject_cloth_prompt:
            cloth_img = cv2.imread(self.args.cloth_image_path)
        else:
            cloth_img = np.zeros((256, 256, 3))
        cloth_prompt_embeds, uncond_cloth_prompt_embeds = self.get_cloth_embeds(
            cloth_image=cloth_img, clip_image_embeds=clip_image_embeds
        )
        with torch.inference_mode():
            # generate text prompt embeds
            prompt_embeds_, negative_prompt_embeds_ = self.basepipe.encode_prompt(
                text_prompt, device=self.args.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)

            """ 4 concatenate multimodal prompt embeds together, format: (text, cloth, face) """
            prompt_embeds = torch.cat([prompt_embeds_, cloth_prompt_embeds], dim=1)
            prompt_embeds = torch.cat([prompt_embeds, face_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_cloth_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_face_prompt_embeds], dim=1)

        """ 5 Using the base pipeline to generate an image based on prompt_embeds. """
        """ The pose image from yolo-nas is passed here for processing. """
        generator = torch.Generator(self.args.device).manual_seed(seed) if seed is not None else None
        image = self.basepipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=pose_img,
            **kwargs,
        ).images

        return image


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
