import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("1.png")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, result = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
contour, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )


for i in range(len(contour)):
    cnt = contour[i]
    # print(hierarchy[i])
    x,y,w,h = cv.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    cv.imwrite(f"data/crop{i}.jpg", crop)
    img = cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255), -1)
    print(cv.contourArea(cnt))
    img = cv.drawContours(img, cnt, -1, (255, 255, 255), cv.FILLED)
    cv.imshow("Image", img)
    cv.imshow("Crop", crop)
    cv.waitKey(0)