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
CONVERT_DIR = path.join(TESTS_DIR, 'convert')

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

def saveImg(name, img):
    save_path = path.join(CONVERT_DIR, name)
    cv2.imwrite(save_path, img)
    return save_path

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
    contours, hierarchy = cv2.findContours(
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

            img = imgOrigin[y:y+h, x:x+w]
            saveImg(name + ".png", img)

def handleImg(img, posX_, posY_, posH_, posW_):
    x, y, c = img.shape
    
    posY = y * (posY_/100)
    posX = x * (posX_/100)
    height = y * (posH_/100)
    width = x * (posW_/100)

    return img[round(posY):round(posY+ height), round(posX):round(posX + width)]

def createImgCMND(img, filename):
    posX_ = 78
    posY_ = 15
    posH_ = 6
    posW_ = 46

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_cmnd.png", img)

def createImgHoten(img, filename):
    posX_ = 70
    posY_ = 21
    posH_ = 8
    posW_ = 82

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_hovaten.png", img)

def createImgNgaysinh(img, filename):
    posX_ = 85
    posY_ = 34
    posH_ = 6
    posW_ = 47

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_ngaysinh.png", img)

def createImgNguyenquan(img, filename):
    posX_ = 88
    posY_ = 40
    posH_ = 6
    posW_ = 41

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_nguyenquan.png", img)

def createImgNgaycapDay(img, filename):
    posX_ = 74
    posY_ = 31
    posH_ = 7
    posW_ = 13

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_ngaycapDay.png", img)

def createImgNgaycapMonth(img, filename):
    posX_ = 100
    posY_ = 31
    posH_ = 7
    posW_ = 13

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_ngaycapMonth.png", img)

def createImgNgaycapYear(img, filename):
    posX_ = 123
    posY_ = 31
    posH_ = 7
    posW_ = 25

    img = handleImg(img, posX_, posY_, posH_, posW_)
    return saveImg(filename + "_ngaycapYear.png", img)

def get_string_from_image(img, type):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    result = 255 - opening
    result = cv2.GaussianBlur(result, (3,3), 0)
    dilation = cv2.dilate(result,kernel,iterations = 1)
    string = image_to_string(dilation, lang=type , config='--psm 6')
    return h(string)

def CMNDFront(file_path, filename):
    img = cv2.imread(file_path)
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = 23
    threshold2 = 155
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, img, filename)
    roiImg = cv2.imread(path.join(CONVERT_DIR, filename + '.png'))

    cmndImg = cv2.imread(createImgCMND(roiImg, filename))
    cmnd = get_string_from_image(cmndImg, 'eng')

    hovatenImg = cv2.imread(createImgHoten(roiImg, filename))
    hovaten = get_string_from_image(hovatenImg, 'vie')

    ngaysinhImg = cv2.imread(createImgNgaysinh(roiImg, filename))
    ngaysinh = get_string_from_image(ngaysinhImg, 'eng')

    nguyenquanImg = cv2.imread(createImgNguyenquan(roiImg, filename))
    nguyenquan = get_string_from_image(nguyenquanImg, 'vie')

    return cmnd, hovaten, ngaysinh, nguyenquan

def CMNDBack(file_path, filename):
    img = cv2.imread(file_path)
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = 23
    threshold2 = 155
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, img, filename)
    roiImg = cv2.imread(path.join(CONVERT_DIR, filename + '.png'))
    
    ngaycapDayImg = cv2.imread(createImgNgaycapDay(roiImg, filename))
    day = get_string_from_image(ngaycapDayImg, 'eng')

    ngaycapMonthImg = cv2.imread(createImgNgaycapMonth(roiImg, filename))
    month = get_string_from_image(ngaycapMonthImg, 'eng')

    ngaycapYearImg = cv2.imread(createImgNgaycapYear(roiImg, filename))
    year = get_string_from_image(ngaycapYearImg, 'eng')

    ngaycap = day+'-'+month+'-'+year
    return ngaycap

