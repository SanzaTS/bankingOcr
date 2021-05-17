import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUi
import pytesseract
import cv2
import numpy as np
import os
import mysql.connector
import random

pytesseract.pytesseract.tesseract_cmd = r'E:\Tessarect\tesseract.exe'

per = 25
pixelThreshold = 500
# coordinates for form to extract data
roi = [[(641, 379), (1117, 400), 'text', 'fName'],
       [(641, 401), (1117, 433), 'text', 'SName'],
       [(640, 438), (1117, 462), 'text', 'LName'],
       [(640, 470), (1117, 492), 'text', 'id'],
       [(640, 587), (1117, 616), 'text', 'strName'],
       [(640, 620), (1117, 646), 'text', 'City'],
       [(640, 642), (1117, 666), 'text', 'Province'],
       [(640, 674), (1117, 706), 'text', 'Code'],
       [(640, 787), (1117, 820), 'text', 'Email'],
       [(640, 824), (1117, 851), 'text', 'Phone']
       ]

roi2 = [
    [(134, 192), (270, 214), 'text', 'idCard']
]


# Main Gui ( Create User)
class CreateUser(QDialog):

    def __init__(self):
        super(CreateUser, self).__init__()
        loadUi("login.ui", self)
        self.btnLogin.clicked.connect(self.saveToDB)
        self.createAcc.clicked.connect(self.goToData)
        self.btnRead.clicked.connect(self.addImage)

    def saveToDB(self):
        fname = self.fName.text()
        sName = self.sName.text()
        lNmae = self.lName.text()
        id = self.id.text()
        addr = self.strName.text()
        city = self.city.text()
        province = self.province.text()
        code = self.code.text()
        email = self.email.text()
        phone = self.phone.text()
        account_name = self.account.currentText()
        account_number = ""

        # Randomizing Account Number
        word = "0123456789"
        charlst = list(word)  # convert the string into a list of characters
        random.shuffle(charlst)  # shuffle the list of characters randomly
        account_number = ''.join(charlst)

        conn = mysql.connector.connect(
            user='root', password=' ', host='127.0.0.1', database='bankOcr')
        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()
        # Preparing SQL query to INSERT a record into the database.
        sql = """INSERT INTO customer(id, F_Name, S_Name, L_Name, Str_Name, City, province, code,email,phone) 
              VALUES (%s, %s, %s, %s, %s, %s ,%s ,%s,%s,%s)"""
        tuples = (id, fname, sName, lNmae, addr, city, province, code, email, phone)

        sql2 = """INSERT INTO account(Account_Number, Name, idNum)
                  VALUES (%s, %s, %s) """
        tuples2 = (account_number, account_name, id)
        # Executing the SQL command
        cursor.execute(sql, tuples)
        cursor.execute(sql2, tuples2)
        # Commit your changes in the database
        conn.commit()
        # Closing the connection
        conn.close()
        QMessageBox.about(self, "Confirmation", "ADone")
        self.clearBox()

    def clearBox(self):
        self.fName.clear()
        self.sName.clear()
        self.lName.clear()
        self.id.clear()
        self.strName.clear()
        self.city.clear()
        self.province.clear()
        self.code.clear()
        self.email.clear()
        self.phone.clear()

    # function to read form and extract data to save to db
    def addImage(self):
        imgQ = cv2.imread('bank.jpg')
        h, w, c = imgQ.shape
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        imKp1 = cv2.drawKeypoints(imgQ, kp1, None)
        # cv2.imshow('Output Image 2.0', imKp1)
        path = 'img'
        myFolder = os.listdir(path)
        # print(myFolder)
        for j, y in enumerate(myFolder):
            img = cv2.imread(path + "/" + y)
            cv2.imshow('image ** ' + y, img)
            kp2, des2 = orb.detectAndCompute(img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des2, des1)
            matches.sort(key=lambda x: x.distance)
            good = matches[:int(len(matches) * (per / 100))]
            imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
            # cv2.imshow('output loop ' + y, imgMatch)

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
                    signed = ''
                    if totalPixel > pixelThreshold:
                        totalPixel = 1
                    else:
                        totalPixel = 0
                    print(f'{r[3]}:  {totalPixel} ')
                    if totalPixel == 1:
                        signed = 'Y'
                    else:
                        signed = 'N'
                    # print('Box is signed(Y/N) : ' + signed)
                    # myData.append(totalPixel)
                    myData.append(signed)

        QMessageBox.about(self, "Confirmation", "Adding image")
        self.fName.setText(myData[0])
        self.sName.setText(myData[1])
        self.lName.setText(myData[2])
        self.id.setText(myData[3])
        self.strName.setText(myData[4])
        self.city.setText(myData[5])
        self.province.setText(myData[6])
        self.code.setText(myData[7])
        self.email.setText(myData[8])
        self.phone.setText(myData[9])

        # cv2.imshow('Output Image', imgQ)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def goToData(self):

        createacc = ReadData()
        widget.addWidget(createacc)
        widget.setCurrentIndex(widget.currentIndex() + 1)


# gui cass
class ReadData(QDialog):
    # initalizing gui
    def __init__(self):
        super(ReadData, self).__init__()
        loadUi("createAcc.ui", self)
        self.btnSignUp.clicked.connect(self.readDataa)
        self.btnId.clicked.connect(self.readId)
        self.homeBtn.clicked.connect(self.goToHome)

    def goToHome(self):
        customer = CreateUser()
        widget.addWidget(customer)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def readId(self):
        imgQ = cv2.imread('id.jpg')
        h, w, c = imgQ.shape
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(imgQ, None)
        imKp1 = cv2.drawKeypoints(imgQ, kp1, None)
        # cv2.imshow('Output Image 2.0', imKp1)
        path = 'id'
        myFolder = os.listdir(path)
        # print(myFolder)
        for j, y in enumerate(myFolder):
            img = cv2.imread(path + "/" + y)
            # cv2.imshow('image ** ' + y, img)
            kp2, des2 = orb.detectAndCompute(img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(des2, des1)
            matches.sort(key=lambda x: x.distance)
            good = matches[:int(len(matches) * (per / 100))]
            imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
            # cv2.imshow('output loop ' + y, imgMatch)

            srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
            imgScan = cv2.warpPerspective(img, M, (w, h))
            # cv2.imshow('output SCAN' + y, imgScan)
            imgShow = imgScan.copy()
            imgMask = np.zeros_like(imgShow)
            myData = []
            print(f'E##### xtracting Data from Form {j}')

            for x, r in enumerate(roi2):
                cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255.0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
                # imGray = cv2.cvtColor(imgScan, cv2.COLOR_BGR2GRAY)
                # imThrash = cv2.threshold(imGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                # cv2.imshow(str(x), imgCrop)
                if r[2] == 'text':
                    # print(f'{r[3]}:  {pytesseract.image_to_string(imgCrop)} ')
                    myData.append(pytesseract.image_to_string(imgCrop))

                if r[2] == 'box':
                    imGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                    imThrash = cv2.threshold(imGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                    totalPixel = cv2.countNonZero(imThrash)
                    signed = ''
                    if totalPixel > pixelThreshold:
                        totalPixel = 1
                    else:
                        totalPixel = 0
                    # print(f'{r[3]}:  {totalPixel} ')
                    if totalPixel == 1:
                        signed = 'Y'
                    else:
                        signed = 'N'
                    # print('Box is signed(Y/N) : ' + signed)
                    # myData.append(totalPixel)
                    myData.append(signed)

                QMessageBox.about(self, "Confirmation", "Adding search image")
                self.id.setText(myData[0])

        # cv2.imshow('Output Image', imgQ)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # function to read and display  data
    def readDataa(self):

        conn = mysql.connector.connect(
            user='root', password=' ', host='127.0.0.1', database='bankOcr')

        id = (self.id.text(),)

        # reading customer table
        query1 = "SELECT * FROM customer WHERE id = %s"
        # query1 = "SELECT * FROM customer "
        mycursor = conn.cursor()

        mycursor.execute(query1, id)

        result = mycursor.fetchall()
        self.customer.setRowCount(0)
        for row_number, row_data in enumerate(result):
            print(row_number)
            self.customer.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                # print(column_number)
                self.customer.setItem(row_number, column_number, QTableWidgetItem(str(data)))

        # Reading Accounts table
        # query2 = "SELECT * FROM account "
        query2 = "SELECT * FROM account WHERE idNum = %s"
        mycursor = conn.cursor()

        mycursor.execute(query2, id)

        result = mycursor.fetchall()
        self.account.setRowCount(0)
        # print(count)
        for row_number, row_data in enumerate(result):
            # print(row_number)
            self.account.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                # print(column_number)
                self.account.setItem(row_number, column_number, QTableWidgetItem(str(data)))


# Running GUI Application
app = QApplication(sys.argv)
mainWindow = CreateUser()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedWidth(1000)
widget.setFixedHeight(700)
widget.show()
app.exec_()
