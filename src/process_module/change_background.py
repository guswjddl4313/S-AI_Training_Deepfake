from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from prompt_random import Rand_prompt
import torch
import os

#torch.cuda.empty_cache()

class Inpaint_images:
    def __init__(self, prompt_file_path):
        # Stable Diffusion 모델 로드
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")
        self.rand_prompt = Rand_prompt(prompt_file_path)

    # Inpainting 작업 함수
    def inpaint_images_in_dir(self, dir_path, masking_dir, output_dir):
        # Create directories if they don't exist
        # os.makedirs(masking_dir, exist_ok=True)
        # os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(dir_path, exist_ok=True)
        # List all files in the dir
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print("No images found in dir.")
            return

        for image_file in image_files:
            image_path = os.path.join(dir_path, image_file)
            mask_image_path = os.path.join(masking_dir, f'masking_{image_file}')

            if not os.path.exists(mask_image_path):
                print(f"Mask image {mask_image_path} not found, skipping.")
                continue

            # 원본 이미지 사이즈 저장
            origin_width, origin_height = Image.open(image_path).size
            print(f"Processing {image_file}: width : {origin_width} / height : {origin_height}")

            # 로컬 이미지를 불러와서 사이즈 조정
            image = Image.open(image_path).resize((1024, 1024))
            mask_image = Image.open(mask_image_path).resize((1024, 1024))

            # 랜덤 시드 생성기 설정
            generator = torch.Generator(device="cuda").manual_seed(0)

            # 랜덤 프롬프트 생성
            prompt = self.rand_prompt.pick()
            print(f"prompt : {prompt}")

            # Inpainting 수행
            inpainted_image = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                guidance_scale=8.0,
                num_inference_steps=15,  # 15에서 30 사이의 값이 적합
                strength=0.99,  # strength는 1.0 이하로 설정
                generator=generator,
            ).images[0]

            # 원본 사이즈로 조정 후 저장
            origin_size_image = inpainted_image.resize((origin_width, origin_height), Image.LANCZOS)
            output_image_path = os.path.join(output_dir, f'inpainted_{image_file}')
            origin_size_image.save(output_image_path)
            print(f"Result saved as {output_image_path}")
