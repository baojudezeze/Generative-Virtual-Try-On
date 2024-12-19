# Introduction

🎨 We propose an end-to-end multi-concept customization method, based on our RMP-Adapter pipeline: A Region-based Multiple Prompt Adapter for Multi-Concept Customization in Text-to-Image Diffusion Model.

✨ What makes it cool? Our method leverages multiple prompt adapters (MP-Adapter) to extract pixel-level information from reference images of target concepts. Through our region-based denoising framework (RDF), we can precisely control where and how different concepts appear in the generated image.

🚀 Check out our inference code and pre-trained models below. Now you can create amazing mashups of different concepts in your generated images!

# Release Plans
- [x] Inference codes and pretrained weights of RMP-Adapter
- [ ] Further Examples of Concept Customization Implementation
- [ ] Training scripts of RMP-Adapter

#  Examples

## Multi-concept Customization

![Example](./asserts/p3.png)

## Virtual Try-on

![Example](./asserts/p2.png)

## Identity-consistent Story Visualization



# Installation

## Environment Preparation

We Recommend a python version >=3.10 and cuda version =12.1. Then build environment as follows:

```bash
# [Optional] Create your virtual env
conda create -n myenv python==3.10
conda activate myenv

# [Optional] Make sure you have pytorch-gpu
pip3 install torch torchvision torchaudio

# Install requirements with pip
cd {your program path}/RMP-Adapter
pip install -r requirements.txt
```

## Download Weights

**Automatic Downloads:** 
The following base model weights will be downloaded automatically:

`SG161222/Realistic_Vision_V5.1_noVAE`

`stabilityai/sd-vae-ft-mse`

`openai/clip-vit-large-patch14`

`IDEA-Research/grounding-dino-tiny`

`facebook/sam-vit-base`

**Manual Downloads:** 

Download our [weights](https://) of RMP-Adapter, and place these weights under `./pretrained_weights/mp_adapter` directory.

You can freely choose alternative SD-1.5-based models (e.g. Realistic_Vision series or various anime-focused models), as well as more advanced versions of DINO and SAM models (e.g. grounding-dino-base, sam-vit-huge). These alternatives should provide better results.

If you do not have access to Hugging Face, please download these base models manually and place them in the `./pretrained_weights` directory.


# Inference of RMP-Adapter

Use the following command to run inference:

```bash
python inference.py --image_prompt_list "<your image prompt path 1>,<your image prompt path 2>" \
--text_prompt <your text prompt> \
--text_object '<keyword 1>. <keyword 2>. ' \
--image_prompt_type '<type 1>,<type 2>' \
--seg_type sam \
```

## Important Notes:

- The order of images in image_prompt_list should match the order of keywords in text_object.

- In text_object, use keywords to describe corresponding image prompts, separated by '.'. For example, if your image list contains a person and a garment, the text_object should be 'human. shirt. '

- For image_prompt_type, the default is a list of 1's. Use type 2 for character concepts and type 1 for item concepts (1 = object, 2 = human).

- Since our model is based on SD1.5 series, try to use 512*512 size when image distortion appears under 768*768 size generation.

## Example Use Cases:

**Concept Customization:** 

```bash
python inference.py --image_prompt_list "./asserts/f1.png,./asserts/f2.png" --text_prompt 'photo of a man and a woman, upper body portrait, wearing jeans, street background' --text_object 'woman. man. ' --image_prompt_type '2,2' --seg_type 'dino'
```

**Virtual Try-on:**

```bash
python inference.py --image_prompt_list "./asserts/c1.png,./asserts/c2.png" --text_prompt 'half-protait, a woman wearing a shirt and white long skirt, walking on the street.' --text_object 'shirt. skirt. ' --image_prompt_type '1,1' --seg_type 'sam'
```

**Identity-consistent Story Visualization:**

```bash
python inference.py --image_prompt_list "./asserts/c1.png,./asserts/c2.png" --text_prompt 'half-protait, a woman wearing a shirt and white long skirt, walking on the street.' --text_object 'shirt. skirt. ' --image_prompt_type '1,1' --seg_type sam
```


# Combination with Other Plug-in Modules

Our model is designed as a plug-in module, making it compatible with other models. The following example demonstrates integration with ControlNet, which is particularly effective for character generation. This combination allows precise control over character poses while preserving fine-grained features of the target concept.

![Example](./asserts/appendix.png)


# Disclaimer

This project is for academic research purposes only. Users are solely responsible for their use of this model and any content they generate. Please ensure all usage complies with legal and ethical standards.

# References

```text
SAM + Stable Diffusion Inpainting: https://colab.research.google.com/drive/1umJUZdqEAcm9GQkzLG-EWXo8_ya-ImRL
OutfitAnyone: https://github.com/HumanAIGC/OutfitAnyone
Dreambooth-Stable-Diffusion: https://github.com/XavierXiao/Dreambooth-Stable-Diffusion
super-gradients: https://github.com/Deci-AI/super-gradients
SAM: https://github.com/facebookresearch/segment-anything
DINO: https://github.com/IDEA-Research/DINO
OMG: https://github.com/kongzhecn/OMG
OpenFlamingo: https://github.com/mlfoundations/open_flamingo
IP-Adapter: https://github.com/tencent-ailab/IP-Adapter
```
