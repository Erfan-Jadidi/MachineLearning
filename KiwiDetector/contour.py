import cv2
import numpy as np
import os

output_path = "F:\\ML\\Project\\KiwiDetector\\rotated_kiwis_with_background"
os.makedirs(output_path, exist_ok=True)

for img_index in range(1, 8):
    image_path = f"F:\\ML\\Project\\KiwiDetector\\kiwi\\{img_index}.jpg"
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([10, 50, 20]) 
    upper_brown = np.array([20, 255, 200]) 

    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour_index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500: 
            x, y, w, h = cv2.boundingRect(contour)

            kiwi_roi = image[y:y+h, x:x+w]

            mask_roi = mask_cleaned[y:y+h, x:x+w]

            for angle in [30, 50, 90, 120]:
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                rotated_kiwi = cv2.warpAffine(kiwi_roi, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
                rotated_mask = cv2.warpAffine(mask_roi, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

                rotated_image = image.copy()
                rotated_mask_inv = cv2.bitwise_not(rotated_mask)

                background = cv2.bitwise_and(rotated_image[y:y+h, x:x+w], rotated_image[y:y+h, x:x+w], mask=rotated_mask_inv)

                combined = cv2.add(background, rotated_kiwi)
                rotated_image[y:y+h, x:x+w] = combined

                output_filename = os.path.join(output_path, f"kiwi_{img_index}_contour_{contour_index}_angle_{angle}.jpg")
                cv2.imwrite(output_filename, rotated_image)

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"Kiwi {contour_index+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detected Kiwis", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
# import cv2, numpy
# image_path = "F:\\ML\\Project\\KiwiDetector\\kiwi\\1.jpg"
# img = cv2.imread(image_path)
# img = img / 255.0
# cv2.imshow("1", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img)
# print(img.shape)

# print(img.flatten())
# print(img / 255.0)