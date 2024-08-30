# Deepfake Detector with Automatic Learning

## Overview
This project aims to develop a deepfake detection model with automated learning capabilities.  
**Project Duration:** 24.08.27 ~ 24.08.30

## Example
### Prompt
```prompt
A stunning tropical beach with crystal-clear turquoise water, 
palm trees swaying in the breeze, and a bright blue sky with soft white clouds.
```
### image
<p align="center">
   <img src="https://github.com/user-attachments/assets/26de7784-4ba9-44c5-85d2-2cc5bab0a2f7" width="45%" height="100%">
   <img src="https://github.com/user-attachments/assets/0e6b6988-52ef-44de-8bbb-0a1925d4053d" width="45%" height="100%"> <br>
   origin → after_process
</p>

## Usage
### Additional necessity for *LINUX*
```bash
# install chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb

# install chromedriver & basic setting
wget -N https://storage.googleapis.com/chrome-for-testing-public/128.0.6570.0/linux64/chromedriver-linux64.zip
unzip chromedriver_linux64.zip
chmod +x chromedriver
sudo mv -f chromedriver /usr/local/share/chromedriver
sudo ln -s /usr/local/share/chromedriver /usr/local/bin/chromedriver
sudo ln -s /usr/local/share/chromedriver /usr/bin/chromedriver
```
### To perform tasks
```bash
# Clone the repository
git clone https://github.com/guswjddl4313/S-AI_Training_Deepfake.git
cd S-AI_Training_Deepfake

# Install required packages
pip install -r requirements.txt

# Run pixel crawler
python pixels_crawler.py

# Process data
python process.py

# Train the model
python train_model.py
```
### After TODO
```bash
# Clone the repository
git clone https://github.com/guswjddl4313/S-AI_Training_Deepfake.git
cd S-AI_Training_Deepfake

# Install required packages
pip install -r requirements.txt

# Run the main script
python run.py
```

## Architecture
1. **Crawl 1000 face images using the Pixels crawler.**
2. **Generate deepfake images:**
   - Randomly split images. (Real/Fake)
   - Detect faces in images using MediaPipe.
   - Generate face masking using MediaPipe.
   - Select prompts randomly. (feat.Chat-GPT4o)
   - Apply inpainting.
3. **Randomly split the dataset (train, val, test).**
4. **Fine-tune a pre-trained deep learning model.**

## TODO
- Implement background crawling and automatically call `process.py` after every 1000 images are collected.
- Pay attention when using `train_model.py`.

## Reference
### Model
#### Pre-trained Deepfake Model
- [Kaggle: Deepfake vs Real Faces Detection (ViT)](https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit)
- [Hugging Face: Deepfake vs Real Image Detection](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
#### Inpainting Model
- [Hugging Face: Stable Diffusion 2 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
### Prompt
- Generated using Chat-GPT4o

## Contributor

+ platypus ([@kimyebean](https://github.com/kimyebean))
+ tyojong ([@guswjddl4313](https://github.com/guswjddl4313))

