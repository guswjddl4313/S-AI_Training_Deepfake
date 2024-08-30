import os
import random
import shutil


class Rand_images:
    def move_random_images(source_dir, destination_dir):
        # Create the source directory if it doesn't exist
        # os.makedirs(source_dir, exist_ok=True)

        # List all image files in the source directory
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print("No images found in source directory.")
            return

        # Shuffle and select half of the images
        random.shuffle(image_files)
        num_images_to_move = len(image_files) // 2
        images_to_move = image_files[:num_images_to_move]

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Move selected images to the destination directory
        for image_file in images_to_move:
            source_path = os.path.join(source_dir, image_file)
            destination_path = os.path.join(destination_dir, image_file)
            shutil.move(source_path, destination_path)
            print(f"Moved {image_file} to {destination_dir}")

        # Rename the source directory to './Real' after moving the files
        new_source_dir = './Dataset/Real'
        if not os.path.exists(new_source_dir):
            os.rename(source_dir, new_source_dir)
            print(f"Renamed {source_dir} to {new_source_dir}")
        else:
            print(f"The directory {new_source_dir} already exists. Cannot rename {source_dir}.")
