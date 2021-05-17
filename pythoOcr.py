import pytesseract
import cv2
import numpy
import os

pytesseract.pytesseract.tesseract_cmd = r'D:\Tessarect\tesseract.exe'



per = 25
roi = [[50, 60, 300, 50], 'box', 'sign']
img = cv2.imread(r'Query.png')
#mg1 = cv2.resize(img, (780, 540), interpolation=cv2.INTER_NEAREST)
h, w, c = img.shape

orb = cv2.ORB_create(1000)

kp1, des1 = orb.detectAndCompute(img1, None)

# img2 = cv2.drawKeypoints(img1, kp1, None)

# cv2.imshow('KeyPoints', img2)

path = 'images'
myFolder = os.listdir(path)
# print(myFolder)

for j, y in enumerate(myFolder):
    images = cv2.imread(path + "/" + y)
    images = cv2.resize(img, (780, 540), interpolation=cv2.INTER_NEAREST)
    #  cv2.imshow('imagge'+y,images)

    kp2, des2 = orb.detectAndCompute(images, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    match = bf.match(des2, des1)

    match.sort(key=lambda x: x.distance)

    good = match[:int(len(match) * (per / 100))]
    imgMatch = cv2.drawMatches(images, kp2, img1, kp1, good[:100], None, flags=2)
    cv2.imshow('output' + y, images)

    srcPoints = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

# imgScan = cv2.warpPerspective(images,M,(w,h))
# imgScan = cv2.resize(img, (780, 540), interpolation=cv2.INTER_NEAREST)

cv2.imshow('Output', imgMatch)
cv2.waitKey(0)
