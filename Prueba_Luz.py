import numpy as np
from djitellopy import tello
import cv2
from time import sleep
import KeyPressModule as kp
import math

######## PARAMETERS ###########

#fSpeed = 117 / 10  # Forward Speed in cm/s   (15cm/s)
fSpeed = 15
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

###############################################

x, y = 500, 500
a = 0
z=0
yaw = 0
kp.init()


me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

cap = cv2.VideoCapture(1)
# hsvVals = [98,0,0,155,106,255]
# hsvVals = [4,0,32,133,63,118]
hsvVals2 = [0,22,73,67,255,255]
hsvVals = [5,34,180,47,82,255]
sensors = (2, 3)  # 2 rows, 3 columns
threshold = 0.2
width, height = 480, 360
senstivity = 3  # if number is high less sensitive
#weights = [-25, -15, 0, 15, 25]
weights = [-35, -25, 0, 25, 35]


curve = 0
modo = 1
pc= 500
def thresholding(img):
    global modo, pc
    if pc <= 300:
        modo=modo+1
        if modo==3:
            modo=1

    if  modo==1 :
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
        upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
        mask = cv2.inRange(hsv, lower, upper)
        pc = cv2.countNonZero(mask)
    if modo==2 :
        lower = np.array([hsvVals2[0], hsvVals2[1], hsvVals2[2]])
        upper = np.array([hsvVals2[3], hsvVals2[4], hsvVals2[5]])
        mask = cv2.inRange(hsv, lower, upper)
        pc = cv2.countNonZero(mask)
        print("Cambio")

        x=x+1
    print("blancos",pc)
    tp = hsv.shape[1] * hsv.shape[0]
    print(tp)
    n = tp*0.95
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
    #print(senOut)
    return senOut

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

    me.send_rc_control(lr, fSpeed, 0, curve)

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 15
    aspeed = 50
    global x, y, yaw, a, z
    d = 0
    z = 0
    if kp.getKey("LEFT"):
        lr = -speed
        d = dInterval
        a = -180
    elif kp.getKey("RIGHT"):
        lr = speed
        d = -dInterval
        a = 180
    if kp.getKey("UP"):
        fb = speed
        d = dInterval
        a = 270
    elif kp.getKey("DOWN"):
        fb = -speed
        d = -dInterval
        a = -90
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -aspeed
        yaw -= aInterval
    elif kp.getKey("d"):
        yv = aspeed
        yaw += aInterval

    if kp.getKey("q"): me.land(); sleep(3)
    if kp.getKey("e"): me.takeoff()

    if kp.getKey("f"):
        z=1

    sleep(interval)
    a += yaw
    x += int(d * math.cos(math.radians(a)))
    y += int(d * math.sin(math.radians(a)))

    return [lr, fb, ud, yv, x, y, z]

while True:
    vals = getKeyboardInput()
    if vals[6] == 1:

        break
    else:
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

while True:
    kp.quit()
    img = me.get_frame_read().frame
    img = np.array_split(img, 2)
    img = img[1]
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)
    imgThres = thresholding(img)
    cx = getContours(imgThres, img)  ## For Translation
    senOut = getSensorOutput(imgThres, sensors)  ## Rotation
    sendCommands(senOut, cx)
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # added waitKey to display images and break out of the loop
        break
