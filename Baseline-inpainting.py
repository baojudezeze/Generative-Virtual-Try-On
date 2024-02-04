import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from matplotlib import pyplot as plt
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

# 1 Setting Up the Stable Diffusion Pipeline
model_dir = "stabilityai/stable-diffusion-2-inpainting"
scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

# 2 load inpainting module
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16",
                                                      torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# 3 image preprocessing
source_image = Image.open("./images/0model.jpg")
target_width, target_height = 512, 512
width, height = source_image.size
if width > height:
    left = (width - height) // 2
    right = width - left
    top = 0
    bottom = height
    source_image = source_image.crop((left, top, right, bottom))
else:
    top = (height - width) // 2
    bottom = height - top
    left = 0
    right = width
    source_image = source_image.crop((left, top, right, bottom))
source_image = source_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
segmentation_image = np.asarray(source_image)

# 4 conduct SAM module
device = torch.device("cuda")
sam = sam_model_registry['vit_h'](checkpoint='./models/sam_vit_h_4b8939.pth')
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(segmentation_image)

input_box = np.array([145, 140, 379, 330])
input_point = np.array([[279, 398]])
input_label = np.array([1])

# 5 Mask region
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

# 6 Show Mask
segmentation_mask = masks[0]
stable_diffusion_mask = Image.fromarray(segmentation_mask)
plt.imshow(segmentation_image, cmap='gray')
plt.show()
plt.imshow(stable_diffusion_mask, cmap='gray')
plt.show()

# 7 generate prompted image
inpainting_prompts = "a picture of a man wearing a patterned shirt."
generator = torch.Generator(device="cuda").manual_seed(10)
encoded_images = []
image = pipe(prompt=inpainting_prompts, guidance_scale=7.5, num_inference_steps=50, generator=generator,
             image=source_image, mask_image=stable_diffusion_mask).images[0]
encoded_images.append(image)
plt.imshow(image, cmap='gray')
plt.show()
image.save('save.png', 'PNG')

