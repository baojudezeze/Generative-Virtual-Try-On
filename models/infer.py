import argparse
import torch
import os
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
from models.pipeline_rdf import RegionBasedDenoisingPipeline
from diffusers import StableDiffusionPipeline
from models.rmp_adapter import RmpAdapterPipeline


def main(args):

    scheduler = DDIMScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(
        dtype=torch.float32, device=args.device)

    latent_processor = AutoProcessor.from_pretrained(args.dino_model_path)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.dino_model_path).to(device=args.device)
    sam_processor = AutoProcessor.from_pretrained(args.sam_model_path)
    sam = AutoModelForMaskGeneration.from_pretrained(args.sam_model_path).to(device=args.device)
    

    # 预留给controlnet
    # controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
    # controlnet = ControlNetModel.from_pretrained(
    #     controlnet_model_path, torch_dtype=torch.float16)

    base_pipe = RegionBasedDenoisingPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=args.dtype,
        scheduler=scheduler,
        vae=vae,
        latent_processor=latent_processor,
        sam_processor=sam_processor,
        dino_model=dino,
        sam_model=sam,
        feature_extractor=None,
        safety_checker=None
    ).to(args.device)

    rmp_adapter_pipe = RmpAdapterPipeline(
        base_pipe,
        args.clip_model_path,
        args.adapter_model_path,
        args.image_prompt_list,
        args.image_prompt_type,
        args.device,
        args.dtype)

    # images = rmp_adapter_pipeline.generate(cloth_image=hidden2,
    #                                        pil_image2=faceimg2,
    #                                        reference_image=hidden3,
    #                                        reference_image2=hidden4,
    #                                        pil_image=faceimg,
    #                                        num_samples=1,
    #                                        image=openpose_image,
    #                                        prompt=ppt,
    #                                        scale=0.8,
    #                                        guidance_scale=7.5,
    #                                        width=768,
    #                                        height=768,
    #                                        stage=1,
    #                                        num_inference_steps=17,
    #                                        seed=42)
    images = rmp_adapter_pipe(
        prompt=args.text_prompt,
        object=args.text_object,
        pixel_prompts=args.image_prompt_list,
        num_inference_steps=args.infernece_steps,
        dtype=args.dtype,
        seed=42,
        width=768,
        height=768)
    image = images[0]

    from matplotlib import pyplot as plt
    image.save('out_1.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":

    # environment setting
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.set_device(3)

    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter_model_path",
                        default="/data/zeyu/RMP-Adapter/tmp_weights.bin", type=str)
    parser.add_argument("--base_model_path", default="/data/zeyu/RMP-Adapter/pretrained_weights/Realistic_Vision_V5.1_noVAE", type=str,  # "stable-diffusion-v1-5/stable-diffusion-v1-5"
                        help="we recommend using more advanced models like SG161222/ series for realistic image generation.")  # "SG161222/Realistic_Vision_V5.1_noVAE"
    parser.add_argument("--vae_model_path",
                        default="/data/zeyu/RMP-Adapter/pretrained_weights/sd-vae-ft-mse", type=str)  # stabilityai/sd-vae-ft-mse
    parser.add_argument("--clip_model_path",
                        default="/data/zeyu/RMP-Adapter/pretrained_weights/mp_adapter/image_encoder", type=str)  # "openai/clip-vit-large-patch14"
    parser.add_argument("--dino_model_path",
                        default="/data/zeyu/RMP-Adapter/pretrained_weights/grounding-dino-tiny", type=str)  # IDEA-Research/grounding-dino-tiny
    parser.add_argument("--sam_model_path",
                        default="/data/zeyu/RMP-Adapter/pretrained_weights/sam-vit-base", type=str)  # facebook/sam-vit-base
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default=torch.bfloat16,
                        help="dtype for model inference")
    parser.add_argument("--image_prompt_list", default=["./images/img1.png", "./images/img3.png"], type=lambda x: x.split(
        ","), help="list of injected image features")  # "prompt1,prompt2,prompt3"
    parser.add_argument("--image_prompt_type", default=[2, 1], type=lambda x: x.split(","),
                        help="list of feature types to inject (1=object, 2=human), must match length of image_prompt_list")  # "1,2,1" None
    parser.add_argument("--infernece_steps", default=50, type=int)  # 50

    parser.add_argument("--text_object", default="face. guitar.",
                        type=str)  # "str"
    parser.add_argument("--text_prompt", default="photo of a young woman, with a guitar at her side, campus baskground.",
                        type=str)  # "str"

    args = parser.parse_args()
    main(args)
