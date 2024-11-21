from skimage.feature import hog
from skimage.color import rgb2gray
import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # تغییر اندازه تصویر
    gray_image = rgb2gray(image)  # تبدیل به سیاه‌وسفید
    features, hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )
    return features

# استخراج ویژگی‌ها از تمامی تصاویر
data = []
labels = []
dataset_path = "F:\\ML\\Project\\KiwiDetector\\kiwi\\"
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_path, filename)
        features = extract_hog_features(image_path)
        data.append(features)
        labels.append(1)

non_kiwi_dataset_path = "F:\\ML\\Project\\KiwiDetector\\Non_kiwi\\"

for filename in os.listdir(non_kiwi_dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(non_kiwi_dataset_path, filename)
        features = extract_hog_features(image_path)
        data.append(features)
        labels.append(0)
        

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# آموزش مدل SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# ارزیابی مدل
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import pickle

with open("F:\ML\Project\KiwiDetector\model.dat", "wb") as file:
    pickle.dump(model, file)

