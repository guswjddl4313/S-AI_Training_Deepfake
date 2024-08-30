# import os
# import random
# from sklearn.model_selection import train_test_split
# from datasets import Dataset, DatasetDict
# from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
# from torchvision.transforms import Compose, Resize, Normalize, ToTensor
# from PIL import Image
# import evaluate

# # Set paths
# fake_dir = './Dataset/Fake'
# real_dir = './Dataset/Real'

# # Load images and labels
# def load_images_from_folder(folder, label):
#     images = []
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#         image = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB mode
#         if image is not None:
#             images.append({"image": image, "label": label})
#     return images

# fake_images = load_images_from_folder(fake_dir, 0)  # 0 for Fake
# real_images = load_images_from_folder(real_dir, 1)  # 1 for Real

# # Combine and shuffle the dataset
# all_images = fake_images + real_images
# random.shuffle(all_images)

# # Split data into train, validation, and test
# train_val_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)
# train_images, val_images = train_test_split(train_val_images, test_size=0.25, random_state=42)  # 0.25 of 0.8 is 0.2

# # Convert to Hugging Face Dataset
# def create_dataset(image_list):
#     return Dataset.from_list(image_list)

# train_dataset = create_dataset(train_images)
# val_dataset = create_dataset(val_images)
# test_dataset = create_dataset(test_images)

# # Create DatasetDict
# dataset_dict = DatasetDict({
#     "train": train_dataset,
#     "validation": val_dataset,
#     "test": test_dataset
# })

# # Load the model and processor
# # model_name = "dima806/deepfake_vs_real_image_detection"
# model_name = "./saved_model"
# num_labels = 2  # 2 classes (Fake and Real)
# model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
# processor = ViTImageProcessor.from_pretrained(model_name)

# # Prepare transforms
# size = processor.size["height"]
# normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

# train_transforms = Compose([
#     Resize((size, size)),
#     ToTensor(),
#     normalize,
# ])

# # Apply transforms to each example in the dataset
# def transform(example_batch):
#     # Apply the transformations to each image in the batch
#     example_batch['pixel_values'] = [train_transforms(image) for image in example_batch['image']]
#     return example_batch

# train_dataset = dataset_dict["train"].map(transform, batched=True)
# val_dataset = dataset_dict["validation"].map(transform, batched=True)
# test_dataset = dataset_dict["test"].map(transform, batched=True)

# # Define a function to compute metrics (accuracy)
# accuracy_metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(axis=-1)
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#     return accuracy

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_strategy="epoch"
# )

# # Initialize Trainer with compute_metrics function
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=processor,
#     compute_metrics=compute_metrics  # Pass the compute_metrics function
# )

# # Continue training the model
# trainer.train()

# # Save the trained model (overwrites the existing saved_model)
# trainer.save_model("./saved_model")

# # Evaluate the model on the training, validation, and test datasets
# train_metrics = trainer.evaluate(train_dataset)
# val_metrics = trainer.evaluate(val_dataset)
# test_metrics = trainer.evaluate(test_dataset)

# print("Training metrics:", train_metrics)
# print("Validation metrics:", val_metrics)
# print("Test metrics:", test_metrics)

import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
import evaluate

# Load the split dataset
def load_split_dataset(split_dir):
    def load_images_from_folder(folder, label):
        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path).convert("RGB")  # Ensure all images are in RGB mode
            if image is not None:
                images.append({"image": image, "label": label})
        return images

    train_real = load_images_from_folder(os.path.join(split_dir, 'train/Real'), 1)
    val_real = load_images_from_folder(os.path.join(split_dir, 'val/Real'), 1)
    test_real = load_images_from_folder(os.path.join(split_dir, 'test/Real'), 1)

    train_fake = load_images_from_folder(os.path.join(split_dir, 'train/Fake'), 0)
    val_fake = load_images_from_folder(os.path.join(split_dir, 'val/Fake'), 0)
    test_fake = load_images_from_folder(os.path.join(split_dir, 'test/Fake'), 0)

    train_images = train_real + train_fake
    val_images = val_real + val_fake
    test_images = test_real + test_fake

    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)

    def create_dataset(image_list):
        return Dataset.from_list(image_list)

    train_dataset = create_dataset(train_images)
    val_dataset = create_dataset(val_images)
    test_dataset = create_dataset(test_images)

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

# Define the main function for training
def main():
    # Load the dataset
    dataset_dir = './Dataset/split'
    dataset_dict = load_split_dataset(dataset_dir)

    # Load the model and processor
    model_name = "ViT"  # Specify your model name here or "./saved_model" if continuing training
    num_labels = 2  # 2 classes (Fake and Real)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Prepare transforms
    size = processor.size["height"]
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

    train_transforms = Compose([
        Resize((size, size)),
        ToTensor(),
        normalize,
    ])

    # Apply transforms to each example in the dataset
    def transform(example_batch):
        # Apply the transformations to each image in the batch
        example_batch['pixel_values'] = [train_transforms(image) for image in example_batch['image']]
        return example_batch

    train_dataset = dataset_dict["train"].map(transform, batched=True)
    val_dataset = dataset_dict["validation"].map(transform, batched=True)
    test_dataset = dataset_dict["test"].map(transform, batched=True)

    # Define a function to compute metrics (accuracy)
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch"
    )

    # Initialize Trainer with compute_metrics function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics  # Pass the compute_metrics function
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model("./saved_model")

    # Evaluate the model on the training, validation, and test datasets
    train_metrics = trainer.evaluate(train_dataset)
    val_metrics = trainer.evaluate(val_dataset)
    test_metrics = trainer.evaluate(test_dataset)

    print("Training metrics:", train_metrics)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    main()
