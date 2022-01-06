import cv2
import numpy as np
import time
import os
import HandTracking as ht

folderPath = "Header"
myList = os.listdir(folderPath)
#print(myList)
palette = []
currentColor = (128, 128, 128)
thickness = 25
x_prev, y_prev = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8)

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    palette.append(image)

#print(len(palette))

header = palette[myList.index('eraser.jpg')]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionConfidence=0.85)

while True:
    #Importing image
    ret, img = cap.read()
    #Flip image to invert the inverted image
    img = cv2.flip(img, 1)

    #Finding hand landmark with module
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)

        x_index, y_index = lmList[8][1:3]
        x_middle, y_middle = lmList[12][1:3]


        #Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #If selection mode - Two fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            x_prev, y_prev = 0, 0
            cv2.circle(img, (x_index, y_index), 15, (255, 255, 255), cv2.FILLED)
            print("Select Mode")

            if y_index < 130:
                if 0 < x_index < 164:
                    header = palette[myList.index('eraser.jpg')]
                    currentColor = (128, 128, 128)

                elif 165 < x_index < 369:
                    header = palette[myList.index('black.jpg')]
                    currentColor = (255, 255, 255)

                elif 370 < x_index < 474:
                    header = palette[myList.index('white.jpg')]
                    currentColor = (0, 0, 0)

                elif 475 < x_index < 639:
                    header = palette[myList.index('red.jpg')]
                    currentColor = (22, 22, 255)

                elif 640 < x_index < 799:
                    header = palette[myList.index('orange.jpg')]
                    currentColor = (77, 145, 255)

                elif 800 < x_index < 954:
                    header = palette[myList.index('yellow.jpg')]
                    currentColor = (89, 222, 255)

                elif 955 < x_index < 1109:
                    header = palette[myList.index('green.jpg')]
                    currentColor = (87, 217, 126)

                elif 1110 < x_index < 1280:
                    header = palette[myList.index('blue.jpg')]
                    currentColor = (255, 182, 56)

        # If draw mode - Index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x_index, y_index), 15, currentColor, cv2.FILLED)
            print("Draw Mode")
            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = x_index, y_index

            if currentColor == (128, 128, 128):
                thickness = 25
            else:
                thickness = 15
                cv2.line(img, (x_prev, y_prev), (x_index, y_index), currentColor, thickness)
                cv2.line(canvas, (x_prev, y_prev), (x_index, y_index), currentColor, thickness)

            x_prev, y_prev = x_index, y_index

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)


    #Display the color palette
    img[0: 130, 0:1280] = header

    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)