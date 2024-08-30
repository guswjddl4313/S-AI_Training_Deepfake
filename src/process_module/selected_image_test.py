from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor, pipeline

# Load the image
image_path = "./test2.jpg"  # Update with the correct path if needed
image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

# Display the image
plt.imshow(image)
plt.axis("off")  # Turn off the axis
plt.show()

# Load your trained model
model_path = "./saved_model"  # Update with the correct path if needed
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

# Use the pipeline to classify the image
pipe = pipeline("image-classification", model=model, feature_extractor=processor, device=0)

# Directly pass the PIL image to the pipeline
result = pipe(image)

# Print the classification result
print(result)
