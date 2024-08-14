import numpy as np
from djitellopy import tello
import cv2
import time

######## PARAMETERS ###########
fSpeed = 30
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval
################################
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
cap = cv2.VideoCapture(1)

hsvVals = [
[23, 255, 254, 24, 255, 255],
[21, 255, 253, 22, 255, 255],
[21, 255, 252, 22, 255, 255],
[20, 144, 220, 23, 255, 255],
[21, 209, 252, 23, 253, 255],
[23, 199, 243, 27, 254, 255],
[26, 141, 245, 30, 192, 255],
[25, 181, 250, 27, 206, 255],
[27, 123, 241, 32, 179, 255],
[25, 202, 248, 27, 217, 255],
[25, 194, 247, 27, 220, 255],
[22, 236, 251, 25, 255, 255],
[25, 212, 249, 28, 252, 255],
[25, 243, 248, 27, 255, 255],
[21, 245, 244, 27, 255, 255],
[19, 255, 235, 23, 255, 255],
[27, 163, 245, 30, 182, 255],
[27, 126, 244, 30, 182, 255],
[24, 45, 181, 34, 212, 255],
[25, 170, 233, 27, 220, 255],
[23, 235, 245, 25, 255, 255],
[25, 188, 246, 29, 241, 255],
[19, 217, 241, 27, 255, 255],
[25, 226, 250, 26, 255, 255],
[24, 200, 247, 27, 255, 255],
[25, 219, 245, 27, 255, 255],
[25, 169, 247, 29, 240, 255],
[23, 242, 250, 25, 255, 255],
[21, 249, 252, 22, 255, 255],
[21, 252, 255, 22, 255, 255],
[22, 252, 247, 22, 255, 255],
[21, 230, 245, 23, 255, 255],
[20, 246, 249, 22, 255, 255],
[21, 226, 244, 22, 252, 255],
[23, 197, 247, 24, 250, 255],
[22, 255, 251, 23, 255, 255],
[23, 251, 251, 24, 255, 255],
[23, 254, 237, 24, 255, 255],
[27, 183, 247, 29, 206, 255]


]
#hsvVals2 = [0, 0, 252, 165, 5, 255]
hsvVals2 = [17, 40, 249, 23, 51, 255]
hsvVals2=[0, 13, 211, 178, 22, 229]
sensors = (2, 3)  # 2 rows, 3 columns
threshold = 0.025
width, height = 480, 360
senstivity = 3  # if number is high less sensitive
weights = [-45, -35, 0, 35, 45]

curve = 0
contador = 0
fin = False
Modofin = False
despegue=False
start = 0
global ri_i
global ri
global i
i = 10
ri_i = False
ri = 0
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def thresholding(img, hsv_vals):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_vals[:3])
    upper = np.array(hsv_vals[3:])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def hsv(*args):
    combined_hsv = np.zeros_like(args[0])
    for img in args:
        combined_hsv = np.where(img != 0, img, combined_hsv)
    return combined_hsv

def fill_closed_areas(mask):
    # Encuentra los contornos de los objetos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rellena los contornos cerrados con puntos blancos
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    return filled_mask

def thresholding_f(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals2[0], hsvVals2[1], hsvVals2[2]])
    upper = np.array([hsvVals2[3], hsvVals2[4], hsvVals2[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def getContours(imgThres, img):
    cx = 0
    x1, y1, x2, y2 = 0, 0, 480, 150
    roi = imgThres[y1:y2, x1:x2]
    #imgP = np.array_split(imgThres, 3)
    #imgP = imgP[0]
    #contours, hierarchy = cv2.findContours(imgP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        x += x1
        y += y1

        cx = x + w // 2
        cy = y + h // 2

        # Dibuja el contorno en la imagen original
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        # Dibuja líneas horizontales y verticales desde el punto (cx, cy)
        cv2.line(img, (cx, 0), (cx, img.shape[0]), (0, 0, 255), 2)  # Línea vertical
        cv2.line(img, (0, cy), (img.shape[1], cy), (0, 0, 255), 2)  # Línea horizontal
    return cx

def getSensorOutput(imgThres, sensors, cx):
    imgs = np.array_split(imgThres, sensors[0])
    totalPixels = (imgThres.shape[1] // 12) * imgThres.shape[0]
    senOut = []
    senOut_1 = []
    for row in imgs:
        rowOut = []
        for col in np.hsplit(row, sensors[1]):
            pixelCount = cv2.countNonZero(col)
            if pixelCount > threshold * totalPixels:
                rowOut.append(1)
            else:
                rowOut.append(0)
        senOut.append(rowOut)

    if (cx < width//5):
        senOut_1 = [1,0,0,0,0]
    elif (cx < 2*(width//5)):
        senOut_1 = [0, 1, 0, 0, 0]
    elif (cx < 3*(width//5)):
        senOut_1 = [0, 0, 1, 0, 0]
    elif (cx < 4*(width//5)):
        senOut_1 = [0, 0, 0, 1, 0]
    elif (cx < 5*(width//5)):
        senOut_1 = [0, 0, 0, 0, 1]
    if senOut == [[0, 0, 0], [0, 0, 0]]:
        senOut_1 = [0, 0, 0, 0, 0]
    return senOut_1

def paro(imgThres):
    imgs = np.array_split(imgThres, 2)
    totalPixels2 = (imgThres.shape[1] // 4) * imgThres.shape[0]
    pixelCount2 = 0
    for img in imgs:
        pixelCount2 += cv2.countNonZero(img)
        print(pixelCount2)
    if pixelCount2 > 4000:
        print("Paro")
        mensajeparo = True
    else:
        print("Vuelo")
        mensajeparo = False
    return mensajeparo

def sendCommands(senOut, cx):
    global curve, ri , ri_i ,i
    # Traslación
    lr = (cx - width // 2) // senstivity
    lr = int(np.clip(lr, -15, 15))
    if 2 > lr > -2: lr = 0
    # Rotation
    if   senOut == [1, 0, 0, 0, 0]: curve = weights[0]
    elif senOut == [0, 1, 0, 0, 0]: curve = weights[1]
    elif senOut == [0, 0, 1, 0, 0]: curve = weights[2]
    elif senOut == [0, 0, 0, 1, 0]: curve = weights[3]
    elif senOut == [0, 0, 0, 0, 1]: curve = weights[4]

    # Intersección

    fSpeed = 30
    if senOut == [0,0,0,0,0]:
        me.send_rc_control(0, 10, 0, curve)
    else:
        me.send_rc_control(lr, fSpeed, 0, curve)


# Seguidor de linea
while True:
    frame_read = me.get_frame_read()
    img = frame_read.frame

    if start == 0:
        me.send_rc_control(0, 0, 0, 0)
        me.takeoff()
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        me.send_rc_control(0,0,-20,0)
        time.sleep(8)
        start = 1

    img = np.array_split(img, 2)
    img = img[1]
    img = cv2.flip(img, 0)
    img = cv2.resize(img, (width, height))
    hsv_images = []
    for hsv_vals in hsvVals:
        imgThres = thresholding(img, hsv_vals)
        hsv_image = fill_closed_areas(imgThres)
        hsv_images.append(imgThres)
    combined_hsv = hsv(*hsv_images)
    final = thresholding_f(img)
    modo = paro(final)

    cx = getContours(combined_hsv, img)  ## For Translation-
    print(cx)
    senOut = getSensorOutput(combined_hsv, sensors, cx)  ## Rotation
    sendCommands(senOut, cx)
    stack=stackImages(1.5,([img,combined_hsv]))
    cv2.imshow("Output", stack)

    if modo == True:
        Modofin = True
    if Modofin == True:
        contador = contador + 1
        if contador < 300:
            fin = False
        else:
            fin = True
    if cv2.waitKey(1) & 0xFF == ord('q') or fin == True:
        break



