import os
import random
import shutil
from sklearn.model_selection import train_test_split

def create_dir_structure(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)

def split_data(real_dir, fake_dir, output_dir, test_size=0.2, val_size=0.25):
    # Gather all files
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]

    if len(real_images) == 0 or len(fake_images) == 0:
        raise ValueError("The dataset directories contain no images. Ensure there are images in the Real and Fake directories.")

    # Split the data
    real_train_val, real_test = train_test_split(real_images, test_size=test_size)
    real_train, real_val = train_test_split(real_train_val, test_size=val_size)

    fake_train_val, fake_test = train_test_split(fake_images, test_size=test_size)
    fake_train, fake_val = train_test_split(fake_train_val, test_size=val_size)

    # Ensure no empty splits
    if len(real_train) == 0 or len(fake_train) == 0:
        raise ValueError("The resulting train set is empty. Adjust `test_size` or `val_size`.")

    # Create the output directories
    create_dir_structure(output_dir, ['train/Real', 'val/Real', 'test/Real', 'train/Fake', 'val/Fake', 'test/Fake'])

    # Move the files
    for f in real_train:
        shutil.copy(f, os.path.join(output_dir, 'train/Real'))
    for f in real_val:
        shutil.copy(f, os.path.join(output_dir, 'val/Real'))
    for f in real_test:
        shutil.copy(f, os.path.join(output_dir, 'test/Real'))

    for f in fake_train:
        shutil.copy(f, os.path.join(output_dir, 'train/Fake'))
    for f in fake_val:
        shutil.copy(f, os.path.join(output_dir, 'val/Fake'))
    for f in fake_test:
        shutil.copy(f, os.path.join(output_dir, 'test/Fake'))

    print("Dataset split completed successfully.")

# if __name__ == "__main__":
#     real_dir = './Dataset/Real'
#     fake_dir = './Dataset/Fake'
#     output_dir = './Dataset/split'
    
#     split_data(real_dir, fake_dir, output_dir, test_size=0.1, val_size=0.2)
