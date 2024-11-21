import cv2
import numpy as np
import os

input_path = "F:\\ML\\Project\\KiwiDetector\\Non_kiwi\\"
output_path = "F:\\ML\\Project\\KiwiDetector\\Non_kiwi\\"
os.makedirs(output_path, exist_ok=True)

angles = [30, 60, 90, 120, 150]  # زاویه‌های چرخش
st = 1
for img_index, filename in enumerate(os.listdir(input_path)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_path, filename)
        image = cv2.imread(image_path)

        cnt = 1
        for angle in angles:
            height, width = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

            output_filename = f"{st}-{cnt}.jpg"
            cv2.imwrite(os.path.join(output_path, output_filename), rotated)
            cnt += 1
        st += 1



