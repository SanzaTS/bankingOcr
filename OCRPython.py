import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'E:\Tessarect\tesseract.exe'

per = 25
pixelThreshold = 500

roi = [[(729, 765), (1117, 400), 'text', 'fName'],
       [(641, 401), (1117, 433), 'text', 'SName'],
       [(640, 438), (1117, 462), 'text', 'LName'],
       [(640, 470), (1117, 492), 'text', 'id'],
       [(640, 587), (1117, 616), 'text', 'strName'],
       [(640, 620), (1117, 646), 'text', 'City'],
       [(640, 642), (1117, 666), 'text', 'Province'],
       [(640, 674), (1117, 706), 'text', 'Code']
       ]
imgQ = cv2.imread('bank.jpg')

h, w, c = imgQ.shape
orb = cv2.ORB_create(1000)
# imgQ = cv2.resize(imgQ, (w // 3, h // 3))
kp1, des1 = orb.detectAndCompute(imgQ, None)
imKp1 = cv2.drawKeypoints(imgQ, kp1, None)
# cv2.imshow('Output Image 2.0', imKp1)

path = 'images'
myFolder = os.listdir(path)
# print(myFolder)

for j, y in enumerate(myFolder):
    img = cv2.imread(path + "/" + y)
    # cv2.imshow('image' + y, img)

    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    # cv2.imshow('output' + y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # cv2.imshow('output SCAN' + y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
    myData = []

    print(f'E##### xtracting Data from Form {j}')

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255.0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        cv2.imshow(str(x), imgCrop)

cv2.imshow('Output Image', imgQ)
cv2.waitKey(0)
cv2.destroyAllWindows()
