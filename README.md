# Generative-Virtual-Try-On

We will make an ambitious attempt to build a project that customizes an generative virtual try-on (VTON) based on any text prompt. The project is in progress, but with some impressive results.

![Example](./asserts/p1.png)

## VTON based on Inpainting pipeline

For this project, we have accomplished the baseline implementation, which is based on the inpainting pipeline provided by [stabilityai](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) to achieve personalized generative virtual try on by masking the garment area. The pipeline is resumed from [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) and trained for another 200k steps. Follows the mask-generation strategy presented in [LAMA](https://github.com/advimman/lama).

### Usage

Before running the baseline program, use the [🤗's Diffusers](https://github.com/huggingface/diffusers) library to run Stable Diffusion 2 inpainting in a simple and efficient manner:

```bash
pip install diffusers transformers accelerate scipy safetensors
```

Then run the inpainting pipeline:

```bash
python Baseline_inpainting.py
```
or:

```bash
SAM_+_Stable_Diffusion_Inpainting_Virtue_Try_on.ipynb
```

## Reference
[SAM + Stable Diffusion Inpainting](https://colab.research.google.com/drive/1umJUZdqEAcm9GQkzLG-EWXo8_ya-ImRL)

[OutfitAnyone](https://github.com/HumanAIGC/OutfitAnyone)

[lora](https://github.com/cloneofsimo/lora)

[Dreambooth-Stable-Diffusion](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)

[segment-anything](https://github.com/facebookresearch/segment-anything)

[super-gradients](https://github.com/Deci-AI/super-gradients)




