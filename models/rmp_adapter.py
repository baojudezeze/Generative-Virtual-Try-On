# Adapterd from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
import torch
import torch.nn as nn
from itertools import repeat
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from models.perceiver_resampler import PerceiverResampler, FFNResampler
from models.attention_processor import MPAdapterProcessor, _XFormersAttnProcessor


class RmpAdapterPipeline:
    def __init__(self, base_pipe, clip_tower_path, mp_adapter_path, pixel_prompt_list, pixel_prompt_type, device, dtype):
        self.device = device
        self.dtype = dtype
        self.pixel_prompt_len = len(pixel_prompt_list)

        # Retrieves the type of image prompt, defaults to '1=object' if not specified
        if pixel_prompt_type:
            self.pixel_prompt_type = pixel_prompt_type
        else:
            self.pixel_prompt_type = list(repeat(1, self.pixel_prompt_len))

        self.base_pipe = base_pipe.to(self.device)
        self.clip_tower_path = clip_tower_path
        self.mp_adapter_path = mp_adapter_path

        # Initialize ModuleList containing image embeds categorized by pixel prompt types
        # where each module corresponds to different scenarios
        self.prepare_mp_adapter_image_embeds = nn.ModuleList([
            PerceiverResampler(
                dim=self.base_pipe.unet.config.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=16,
                embedding_dim=1280,  # self.image_encoder.config.hidden_size,
                output_dim=self.base_pipe.unet.config.cross_attention_dim,
                ff_mult=4,
            ).to(self.device, dtype=self.dtype) if prompt_type == 1
            else FFNResampler(
                cross_attention_dim=self.base_pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=1280,  # self.image_encoder.config.hidden_size,
            ).to(self.device, dtype=self.dtype) if prompt_type == 2
            else None
            for prompt_type in self.pixel_prompt_type
        ])

        self.set_mp_adapter_attn()
        self.load_mp_adapter_weights()

    def set_mp_adapter_attn(self):
        attn_procs = {}
        for name in self.base_pipe.unet.attn_processors.keys():

            cross_attention_dim = None if name.endswith("attn1.processor") else self.base_pipe.unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                query_dim = self.base_pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                query_dim = list(reversed(self.base_pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                query_dim = self.base_pipe.unet.config.block_out_channels[block_id]

            if name.endswith("attn1.processor"):
                attn_procs[name] = _XFormersAttnProcessor()
            else:
                attn_procs[name] = MPAdapterProcessor(
                    query_dim=query_dim,
                    cross_attention_dim=cross_attention_dim,
                    added_kv_injt_num=self.pixel_prompt_len,
                    only_text_attention=False if self.pixel_prompt_type else True,  
                    device=self.device,
                    dtype=self.dtype)
        self.base_pipe.unet.set_attn_processor(attn_procs)
        
    def load_mp_adapter_weights(self):
        
        # for key in self.prepare_mp_adapter_image_embeds.state_dict().keys():
        #     print(key)
        
        state_dict = torch.load(self.mp_adapter_path, map_location="cpu")
        perceiver_resampler_dict = state_dict["perceiver_resampler"]
        ffnresampler_dict = state_dict["ffnresampler"]
        mp_16_dict = state_dict["mp_adapter_16"]
        mp_256_dict = state_dict["mp_adapter_256"]
        
        for prompt_type, prepare_mp_adapter_image_embed in zip(self.pixel_prompt_type, self.prepare_mp_adapter_image_embeds):
            if prompt_type == 1:
                prepare_mp_adapter_image_embed.load_state_dict(perceiver_resampler_dict)
            else:
                prepare_mp_adapter_image_embed.load_state_dict(ffnresampler_dict)

        adapter_layers = torch.nn.ModuleList([])
        for key in self.base_pipe.unet.attn_processors:
            if key.find("attn2") > 0:
                adapter_layers.append(nn.ModuleList([self.base_pipe.unet.attn_processors[key]]))
        
        current_state_dict = adapter_layers.state_dict()
        for key in current_state_dict.keys():
            for index, type in enumerate(self.pixel_prompt_type):
                if 'to_mp_k.' + str(index) in key:
                    parts = key.split('.')
                    _key = f'{parts[0]}.{parts[2]}.{parts[4]}'
                    if type == 1:
                        current_state_dict[key] = mp_16_dict[_key]
                    else:
                        current_state_dict[key] = mp_256_dict[_key]
                        
                if 'to_mp_v.' + str(index) in key:
                    parts = key.split('.')
                    _key = f'{parts[0]}.{parts[2]}.{parts[4]}'
                    if type == 1:
                        current_state_dict[key] = mp_16_dict[_key]
                    else:
                        current_state_dict[key] = mp_256_dict[_key]        
                
        adapter_layers.load_state_dict(current_state_dict)

    def clip_tower(self,
                   clip_tower_path,
                   device,
                   dtype,
                   ):
        return (
            CLIPImageProcessor(),
            CLIPVisionModelWithProjection.from_pretrained(clip_tower_path).to(
                device, dtype=dtype
            )
        )

    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image, return_tensors="pt").pixel_values

            #  区别之处
            #  Normal Tensor(1, 1024)，用的是CLIPVisionModelOutput的image_embeds
            clip_image_embeds = self.image_encoder(clip_image.to(
                self.device, dtype=torch.float16)).image_embeds
            t_normal = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float16))
        else:
            clip_image_embeds = clip_image_embeds.to(
                self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        t = torch.zeros_like(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(t)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def __call__(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pixel_prompts: Optional[list] = None,
        object: Optional[str] = None,
        mp_scale: float = 1.0,
        lora_scale: float = 0,
        guidance_scale: int = 7.5,
        do_classifier_free_guidance: Optional[bool] = True,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        image=None,  # for controlnet
        **kwargs,
    ):
        self.check_inputs(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.base_pipe.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

            clip_embeds_list = []
            un_clip_embeds_list = []
            if pixel_prompts is not None:
                for pixel_prompt in pixel_prompts:
                    if isinstance(pixel_prompt, str):
                        pixel_input = Image.open(pixel_prompt)

                        clip_processor, clip_encoder = self.clip_tower(
                            self.clip_tower_path, self.device, self.dtype)
                        clip_embeds = clip_processor(
                            images=pixel_input, return_tensors="pt").pixel_values

                        un_clip_embeds = torch.zeros_like(clip_embeds)
                        clip_embeds = clip_encoder(clip_embeds.to(
                            self.device, dtype=torch.float16), output_hidden_states=True).hidden_states[-2]
                        un_clip_embeds = clip_encoder(un_clip_embeds.to(
                            self.device, dtype=torch.float16), output_hidden_states=True).hidden_states[-2]

                        clip_embeds_list.append(clip_embeds)
                        un_clip_embeds_list.append(un_clip_embeds)

            image_prompt_embeds_list = []
            un_image_prompt_embeds_list = []
            for i, prepare_mp_adapter_image_embed in enumerate(self.prepare_mp_adapter_image_embeds):
                if prepare_mp_adapter_image_embed is not None:
                    image_prompt_embeds_list.append(
                        prepare_mp_adapter_image_embed(clip_embeds_list[i]))
                    un_image_prompt_embeds_list.append(
                        prepare_mp_adapter_image_embed(un_clip_embeds_list[i]))

            generator = torch.Generator(self.device).manual_seed(
                seed) if seed is not None else None

            images = self.base_pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pixel_embeds=image_prompt_embeds_list,
                negative_pixel_embeds=un_image_prompt_embeds_list,
                object = object,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs,
            ).images

        return images
