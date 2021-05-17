import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'E:\Tessarect\tesseract.exe'

per = 25
pixelThreshold = 500

roi = [[(52, 497), (339, 541), 'text', 'Name'],
       [(337, 495), (665, 539), 'text', 'Phone'],
       [(52, 578), (76, 603), 'box', 'Sign'],
       # [(379, 579), (309, 605), 'box', 'Allergy'],
       # [(52, 77), (342, 761), 'text', 'Email'],
       # [(375, 713), (667, 763), 'text', 'ID'],
       # [(52, 804), (341, 847), 'text', 'City'],
       # [(376, 801), (608, 847), 'text', 'Country']
       ]

imgQ = cv2.imread('Query.jpg')
h, w, c = imgQ.shape
orb = cv2.ORB_create(1000)
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
        # cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print(f'{r[3]}:  {pytesseract.image_to_string(imgCrop)} ')
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] == 'box':
            imGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imThrash = cv2.threshold(imGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            totalPixel = cv2.countNonZero(imThrash)
            if totalPixel > pixelThreshold:
                totalPixel = 1
            else:
                totalPixel = 0
            print(f'{r[3]}:  {totalPixel} ')
            myData.append(totalPixel)

    with open('dataOutput.csv', 'a+') as f:
        f.write('\n')
        for data in myData:
            f.write(str(data) + ',')
            f.write('\n')
    cv2.imshow('outputs' + y, imgShow)

cv2.imshow('Output Image', imgQ)
cv2.waitKey(0)
cv2.destroyAllWindows()
