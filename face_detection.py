import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os


class Face_masking:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        #self.mp_drawing = mp.solutions.drawing_utils

    def process_images_in_dir(self, random_dir, output_dir):
        # Create directories if they don't exist
        # os.makedirs(random_dir, exist_ok=True)
        # os.makedirs(output_dir, exist_ok=True)
        # List all files in the dir
        random_image_files = [f for f in os.listdir(random_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not random_image_files:
            print(f"No images found in {random_dir}.")
            return

        # Initialize MediaPipe Face Detection
        with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:  
            for image_file in random_image_files:
                image_path = os.path.join(random_dir, image_file)
                image = cv2.imread(image_path)

                # Convert the image to RGB
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.detections:
                    print(f"Face not found in {image_file}. Moving to {not_face_dir}.")
                    # Move the file to the not_face_dir
                    shutil.move(image_path, os.path.join(not_face_dir, image_file))
                else:
                    print(f'Found {len(results.detections)} faces in {image_file}.')

                    # Create a white background image
                    white_image = np.ones_like(image) * 255

                    for detection in results.detections:
                        # Extract the bounding box of the face
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        x_min = int(bboxC.xmin * iw)
                        y_min = int(bboxC.ymin * ih)
                        x_max = x_min + int(bboxC.width * iw)
                        y_max = y_min + int(bboxC.height * ih)

                        # Draw a black rectangle over the face
                        cv2.rectangle(white_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

                    # Save the result image in the output_dir
                    output_image_path = os.path.join(output_dir, f'masking_{image_file}')
                    cv2.imwrite(output_image_path, white_image)
                    print(f"Result saved as {output_image_path}")
