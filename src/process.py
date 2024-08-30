import os
import shutil
from process_module.image_random import Rand_images
from process_module.face_detection import Face_masking
from process_module.change_background import Inpaint_images
from process_module.split_dataset import split_data

def reset_directory(directory):
    # If the directory exists, remove it and all its contents
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    # Create a fresh directory
    os.makedirs(directory, exist_ok=True)

if __name__ == '__main__':
    try:
        # Usage
        ## Rand_images
        source_dir = './1000_images'
        destination_dir = './process/random'
        
        ## Inpaint_images
        prompt_file_path = './process_module/prompt_dataset.txt' #prompt 데이터 경로
        ii_output_dir = './Dataset/Fake'     # 결과 이미지 저장 경로
        
        ## Inpaint_images & Face_masking
        dir_path = './process/random' # 원본 이미지 경로
        masking_dir = './process/masking'   # 마스크 이미지 경로
        not_face_dir = './not_face' # face_detection에서 얼굴 인식 안 됨

        # After creating Real and Fake directories, split the dataset
        real_dir = './Dataset/Real'
        fake_dir = './Dataset/Fake'
        output_dir = './Dataset/split'
        
        reset_directory(dir_path)
        reset_directory(masking_dir)
        reset_directory(ii_output_dir)
        reset_directory(not_face_dir)
        reset_directory(real_dir)
        reset_directory(output_dir)

        # Active
        face_masking = Face_masking()
        inpaint_images = Inpaint_images(prompt_file_path)

        Rand_images.move_random_images(source_dir, destination_dir)
        face_masking.process_images_in_dir(dir_path, masking_dir)
        inpaint_images.inpaint_images_in_dir(dir_path, masking_dir, ii_output_dir)
        split_data(real_dir, fake_dir, output_dir)

    except Exception as e:
        print(e)
