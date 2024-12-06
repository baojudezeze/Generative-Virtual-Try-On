import argparse
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
from models.pipeline_rdf import RegionBasedDenoisingPipeline
from models.rmp_adapter import RmpAdapterPipeline


def main(args):

    # Load modules will be used in inference
    scheduler = DDIMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(dtype=torch.float32, device=args.device)
    latent_processor = AutoProcessor.from_pretrained(args.dino_model_path)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model_path).to(device=args.device)
    sam_processor = AutoProcessor.from_pretrained(args.sam_model_path)
    sam = AutoModelForMaskGeneration.from_pretrained(args.sam_model_path).to(device=args.device)

    # Load rdf pipeline and mp-adapters
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

    image = rmp_adapter_pipe(
        prompt=args.text_prompt,
        negative_prompt=args.negative_prompt,
        object=args.text_object,
        pixel_prompts=args.image_prompt_list,
        num_inference_steps=args.infernece_steps,
        pixel_embeds_injt_num=args.pixel_embeds_injt_num,
        dtype=args.dtype,
        seed=args.seed,
        width=args.width,
        height=args.height)[0]
    image.save('out_image.png')
  

# if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--height", default=768, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--infernece_steps", default=50, type=int)
    parser.add_argument("--pixel_embeds_injt_num", default=20, type=int,
                        help="the starting step number where image features are injected into the model inference process")
    parser.add_argument("--dtype", default=torch.bfloat16,
                        help="dtype for model inference")
    parser.add_argument("--adapter_model_path",
                        default="./pretrained_weights/mp_adapter/pipeline_weights.bin", type=str)
    parser.add_argument("--base_model_path", default="SG161222/Realistic_Vision_V5.1_noVAE", type=str,
                        help="we recommend using more advanced models like SG161222/* series for realistic image generation.")
    parser.add_argument("--vae_model_path",
                        default="stabilityai/sd-vae-ft-mse", type=str)
    parser.add_argument("--clip_model_path",
                        default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--dino_model_path",
                        default="IDEA-Research/grounding-dino-tiny", type=str)
    parser.add_argument("--sam_model_path",
                        default="facebook/sam-vit-base", type=str)
    parser.add_argument("--image_prompt_list", required=True, type=lambda x: x.split(","),
                        help="list of injected image features, should in format of ['prompt1', 'prompt2', 'prompt3']")
    parser.add_argument("--image_prompt_type", default=None, type=lambda x: x.split(","),
                        help="list of feature types to inject (1=object, 2=human), must match length of image_prompt_list, e.g.:[2,1,1]")
    parser.add_argument("--text_prompt", required=True, type=str,
                        help="Text prompt used during model inference")
    parser.add_argument("--text_object", required=True, type=str,
                        help="Target concepts to be injected into the model, should match the order in image_prompt_list and be separated by '.', e.g.: 'woman. t-shirt. dog.'")
    parser.add_argument(
        "--negative_prompt", default="deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4", type=str)
    args = parser.parse_args()
    main(args)
