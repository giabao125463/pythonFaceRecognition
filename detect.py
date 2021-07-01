import cv2
import numpy as np
from os import path, replace

import pytesseract
from pytesseract import image_to_string

try:
    from PIL import Image
except ImportError:
    import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
frameWidth = 640
frameHeight = 480
TESTS_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.join(TESTS_DIR, 'upload')

test_file = 'cmd.jpg'
test_file_path = path.join(DATA_DIR, test_file)


def empty(a):
    pass

def h(text):
    text = text.replace('\\', '')
    text = text.replace('`', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('>', '')
    text = text.replace('#', '')
    text = text.replace('+', '')
    text = text.replace('.', '')
    text = text.replace('!', '')
    text = text.replace('$', '')
    text = text.replace('„', '')
    text = text.replace(',', '')
    text = text.replace('\'', '')
    text = text.replace('', '')

    return text

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img, imgContour, imgOrigin, name):
    _, contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # minArea = cv2.getTrackbarPos("minArea", "Parameters")
    minArea = 50000
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            roi = imgOrigin[y:y+h, x:x+w]
            cv2.imwrite(name + ".png", roi)

def handleImg(img, posX_, posY_, posH_, posW_):
    x, y, c = img.shape
    
    posY = y * (posY_/100)
    posX = x * (posX_/100)
    height = y * (posH_/100)
    width = x * (posW_/100)

    return img[round(posY):round(posY+ height), round(posX):round(posX + width)]

def createImgCMND(img):
    posX_ = 81
    posY_ = 15
    posH_ = 6
    posW_ = 46

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("cmnd.png", roi)

def createImgHoten(img):
    posX_ = 70
    posY_ = 21
    posH_ = 8
    posW_ = 82

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("hovaten.png", roi)

def createImgNgaysinh(img):
    posX_ = 91
    posY_ = 34
    posH_ = 6
    posW_ = 35

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("ngaysinh.png", roi)

def createImgNguyenquan(img):
    posX_ = 88
    posY_ = 40
    posH_ = 6
    posW_ = 41

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("nguyenquan.png", roi)

def createImgNgaycapDay(img):
    posX_ = 74
    posY_ = 31
    posH_ = 7
    posW_ = 13

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("ngaycapDay.png", roi)

def createImgNgaycapMonth(img):
    posX_ = 100
    posY_ = 31
    posH_ = 7
    posW_ = 13

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("ngaycapMonth.png", roi)

def createImgNgaycapYear(img):
    posX_ = 123
    posY_ = 31
    posH_ = 7
    posW_ = 25

    roi = handleImg(img, posX_, posY_, posH_, posW_)
    cv2.imwrite("ngaycapYear.png", roi)

def get_string_from_image(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    posX_ = cv2.getTrackbarPos("posX_", "Detect")
    posY_ = cv2.getTrackbarPos("posY_", "Detect")
    ret,thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_TRUNC)

    thresh = Image.fromarray(thresh.astype(np.uint8))
    string = image_to_string(thresh, 'vie', config='--psm 6')
    return h(string)

def CMNDFront(file_path):
    img = cv2.imread(file_path)
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = 23
    threshold2 = 155
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, img, 'front')
    roiImg = cv2.imread('front.png')

    createImgCMND(roiImg)
    cmndImg = cv2.imread('cmnd.png')
    cmnd = get_string_from_image(cmndImg)

    createImgHoten(roiImg)
    hovatenImg = cv2.imread('hovaten.png')
    hovaten = get_string_from_image(hovatenImg)

    createImgNgaysinh(roiImg)
    ngaysinhImg = cv2.imread('ngaysinh.png')
    ngaysinh = get_string_from_image(ngaysinhImg)

    createImgNguyenquan(roiImg)
    nguyenquanImg = cv2.imread('nguyenquan.png')
    nguyenquan = get_string_from_image(nguyenquanImg)

    return cmnd, hovaten, ngaysinh, nguyenquan

def CMNDBack(file_path):
    img = cv2.imread(file_path)
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = 23
    threshold2 = 155
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, img, 'back')
    roiImg = cv2.imread('back.png')
    
    createImgNgaycapDay(roiImg)
    ngaycapDayImg = cv2.imread('ngaycapDay.png')
    day = get_string_from_image(ngaycapDayImg)

    createImgNgaycapMonth(roiImg)
    ngaycapMonthImg = cv2.imread('ngaycapMonth.png')
    month = get_string_from_image(ngaycapMonthImg)

    createImgNgaycapYear(roiImg)
    ngaycapYearImg = cv2.imread('ngaycapYear.png')
    year = get_string_from_image(ngaycapYearImg)

    ngaycap = day+'-'+month+'-'+year
    return ngaycap

