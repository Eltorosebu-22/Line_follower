import cv2

# Capturar una imagen de referencia del objeto que deseas detectar
img_ref1 = cv2.imread("C:\\Users\\DELL\\Pictures\\Screenshots\\b.png")
img_ref2 = cv2.imread("C:\\Users\\DELL\\Pictures\\Screenshots\\b1.png")
img_ref3 = cv2.imread("C:\\Users\\DELL\\Pictures\\Screenshots\\n13.png")
img_ref4 = cv2.imread("C:\\Users\\DELL\\Pictures\\Screenshots\\n14.png")
img_ref5 = cv2.imread("C:\\Users\\DELL\\Pictures\\Screenshots\\n15.png")

img = [img_ref1, img_ref2, img_ref3, img_ref4, img_ref5]
for x in img:
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    # Aplicar el método Otsu para encontrar el umbral óptimo
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convertir la imagen original a HSV
    img_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    # Dividir la imagen HSV en tres canales: H, S y V
    h, s, v = cv2.split(img_hsv)

    # Aplicar el umbral óptimo a los canales H, S y V
    h_thresh = cv2.inRange(h, 0, 255)
    s_thresh = cv2.inRange(s, 0, 255)
    v_thresh = cv2.inRange(v, 0, 255)

    # Combinar los canales H, S y V umbralizados en una imagen binaria
    img_bin = cv2.bitwise_and(h_thresh, cv2.bitwise_and(s_thresh, v_thresh))

    # Encontrar el contorno de la figura en la imagen binaria
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el rectángulo delimitador que encierra la figura
    x, y, w, h = cv2.boundingRect(contours[0])

    # Extraer los valores mínimos y máximos de H, S y V del rectángulo delimitador
    h_min, s_min, v_min = img_hsv[y:y + h, x:x + w].min(axis=(0, 1))
    h_max, s_max, v_max = img_hsv[y:y + h, x:x + w].max(axis=(0, 1))

    # Ajustar los valores de lower_range y upper_range utilizando los valores mínimos y máximos de H, S y V
    lower_range = (h_min, s_min, v_min)
    upper_range = (h_max, s_max, v_max)
    h = [h_min, s_min, v_min, h_max, s_max, v_max]
    print(h)
    # Utilizar los valores de lower_range y upper_range para detectar la figura en tiempo real

