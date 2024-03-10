import argparse

import cv2

from model.MultimodalAdapter import MultimodalAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=['cuda', 'cpu'],
    )
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        required=True,
        help="Path of reference image, the best input size is 1024*1024, "
             "please provide an image with a full-body view of a persion without any facial obstruction, "
             "insufficient exposure of the person’s body parts may affect the YOLO-NAS-POSE algorithm’s ability to "
             "accurately generate pose images."
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="The text prompt for generative virtual try-on, if the input is None,"
             "the program will automatically set to: best quality, high quality"
    )
    parser.add_argument(
        "--inject_cloth_prompt",
        type=bool,
        default=False,
        help="Whether to inject the option of cloth prompt, set default as False, "
             "if you want to inject the prompt of target cloth to the model, select True here.",
    )
    parser.add_argument(
        "--cloth_image_path",
        type=str,
        default=None,
        required=False,
        help="Path of cloth image, note that when activating cloth prompt, "
             "there should be no keywords like 'cloth' or 'garment' in the text prompt description, "
             "otherwise, it may cause conflicts in the generated image",
    )
    parser.add_argument(
        "--face_scale",
        type=float,
        default=0.7,
        required=False,
    )
    parser.add_argument(
        "--cloth_scale",
        type=float,
        default=0.6,
        required=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=0,
        required=True,
    )
    parser.add_argument(
        "--yolo_nas_model_type",
        type=str,
        default="yolo_nas_pose_l",
        choices=["yolo_nas_pose_l", "yolo_nas_pose_m", "yolo_nas_pose_s", "yolo_nas_pose_n"],
        help="The model type of yolo-nas-pose, for more details, "
             "see: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md",
    )
    parser.add_argument(
        "--sam_checkpoint_path",
        type=str,
        default="checkpoints/sam_vit_h_4b8939.pth",
        help="The model checkpoint path of SAM segmentation",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="The model type of SAM segmentation, default is vit_h",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default="lllyasviel/control_v11p_sd15_openpose",
        help="The base pipeline of controlnet model, here, the openpose checkpoint is adopted"
    )
    parser.add_argument(
        "--base_sd_model_path",
        type=str,
        default="SG161222/Realistic_Vision_V4.0_noVAE",
        choices=['SG161222/Realistic_Vision_V4.0_noVAE', 'runwayml/stable-diffusion-v1-5'],
        help="The base pipeline of stable diffusion model, default is Realistic_Vision_V4.0_noVAE"
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="checkpoints/image_encoder",
        help="The path to store the image encoder module",
    )
    parser.add_argument(
        "--multimodal_adapter_path",
        type=str,
        default="checkpoints/multimodal-adapter.pth",
        help="The path to store the multimodal adapter module",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # read the reference image
    reference_image = cv2.imread(args.reference_image_path)
    # init pipeline
    pipe = MultimodalAdapter(args=args, num_tokens=257)
    # generate the image
    image = pipe.MultimodalPipe(
        seed=66, ref_img=reference_image,
        face_scale=args.face_scale, cloth_scale=args.cloth_scale,
        width=reference_image.shape[1], height=reference_image.shape[0],
        text_prompt=args.text_prompt,
        num_inference_steps=args.num_inference_steps)[0]
    image.show()
    image.save('output_image.jpg')
    print(1)
