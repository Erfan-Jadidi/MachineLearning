import pickle
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray

with open("F:\ML\Project\KiwiDetector\model.dat", "rb") as file:
    model = pickle.load(file)

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

for i in range(1, 6):
    img_path = f"F:\\ML\\Project\\KiwiDetector\\test\\{i}.jpg"

    features = extract_hog_features(img_path)

    prediction = model.predict_proba([features])
    # print(prediction[0][0])

    print("kiwi" if prediction[0][1] >= 0.8 else "human needed to predict")