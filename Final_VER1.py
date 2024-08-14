import numpy as np
from djitellopy import tello
import cv2
import time
import math

######## PARAMETERS ###########

#fSpeed = 117 / 10  # Forward Speed in cm/s   (15cm/s)
fSpeed = 30
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

###############################################

x, y = 500, 500
a = 0
z=0
yaw = 0

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

cap = cv2.VideoCapture(1)
#hsvVals = [9, 30, 193, 23, 54, 235]
hsvVals = [[20, 30, 144, 27, 58, 177]
    ,[1, 33, 123, 178, 105, 177]
    ,[15, 15, 159, 32, 46, 181]
    ,[19, 18, 155, 36, 51, 169]
    ,[1, 57, 154, 33, 176, 255]]

hsvVals2 = [0, 0, 252, 165, 5, 255]
sensors = (2, 3)  # 2 rows, 3 columns
threshold = 0.025
width, height = 480, 360
senstivity = 3  # if number is high less sensitive
weights = [-45, -25, 0, 25, 45]

curve = 0
contador = 0
fin = False
Modofin = False
def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def thresholding_f(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals2[0], hsvVals2[1], hsvVals2[2]])
    upper = np.array([hsvVals2[3], hsvVals2[4], hsvVals2[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def getContours(imgThres, img):
    cx = 0
    imgP = np.array_split(imgThres, 3)
    imgP = imgP[0]
    contours, hierarchy = cv2.findContours(imgP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # fixed typo
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, [biggest], -1, (255, 0, 255), 7)  # added brackets around biggest
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx

def getSensorOutput(imgThres, sensors):
    imgs = np.array_split(imgThres, sensors[0])
    #totalPixels = (imgThres.shape[1] // sensors[1]) * imgThres.shape[0]
    totalPixels = (imgThres.shape[1] // 12) * imgThres.shape[0]
    senOut = []
    for row in imgs:
        rowOut = []
        for col in np.hsplit(row, sensors[1]):
            pixelCount = cv2.countNonZero(col)
            if pixelCount > threshold * totalPixels:
                rowOut.append(1)
            else:
                rowOut.append(0)
        senOut.append(rowOut)
    print(senOut)
    return senOut

def paro(imgThres):
    imgs = np.array_split(imgThres, 2)
    totalPixels2 = (imgThres.shape[1] // 4) * imgThres.shape[0]
    pixelCount2 = 0
    for img in imgs:
        pixelCount2 += cv2.countNonZero(img)
    if pixelCount2 > 14000:
        print("Paro")
        mensajeparo = True
    else:
        print("Vuelo")
        mensajeparo = False
    return mensajeparo

def sendCommands(senOut, cx):

    global curve

    ## TRANSLATION

    lr = (cx - width // 2) // senstivity

    lr = int(np.clip(lr, -10, 10))

    if 2 > lr > -2: lr = 0

    ## Rotation


    if   senOut[0] == [1, 0, 0]: curve = weights[0]

    elif senOut[0] == [1, 1, 0]: curve = weights[1]

    elif senOut[0] == [0, 1, 0]: curve = weights[2]

    elif senOut[0] == [0, 1, 1]: curve = weights[3]

    elif senOut[0] == [0, 0, 1]: curve = weights[4]

    elif senOut[0] == [0, 0, 0]: curve = weights[2]

    elif senOut[0] == [1, 1, 1]: curve = weights[2]

    elif senOut[0] == [1, 0, 1]: curve = weights[2]

    if  senOut == [[1, 1, 1], [0, 1, 0]]:
        fSpeed = 60
    else:
        fSpeed = 25

    me.send_rc_control(lr, fSpeed, 0, curve)

x=0
## despegue y movilización a altura adecuada
while True:
    me.takeoff()
    time.sleep(4)
    # Mover el dron hacia abajo
    me.move_down(60)  # 50 cm hacia abajo
    time.sleep(2)
    x=1
    if x==1:
        break

while True:

    img = me.get_frame_read().frame
    img = np.array_split(img, 2)
    img = img[1]
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)

    for hsv_vals in hsvVals:
        imgThres = thresholding(img, hsv_vals)
        hsv_image = fill_closed_areas(imgThres)
        hsv_images.append(hsv_image)
    combined_hsv = hsv(*hsv_images)

    final = thresholding_f(img)
    modo = paro(final)
    cx = getContours(combined_hsv, img)  ## For Translation-
    senOut = getSensorOutput(imgThres, sensors)  ## Rotation
    sendCommands(senOut, cx)
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    if modo == True:
        Modofin = True
    if Modofin == True:
        contador = contador + 1
        if contador < 50:
            fin = False
        else:
            fin = True
    if cv2.waitKey(1) & 0xFF == ord('q') or fin == True:  # added waitKey to display images and break out of the loop
        break