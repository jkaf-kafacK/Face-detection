
import numpy as np
import pandas as pd
import cv2

PARTH = "F:/3 ANNO/tirocino/Input_Dataset/62_celebaMask-HQ-img/Train/62_celebaMask-HQ-img_10.jpg"

image = cv2.imread(PARTH, cv2.IMREAD_UNCHANGED)
txt = pd.read_table(
    "F:/3 ANNO/tirocino/Input_Dataset/62_celebaMask-HQ-img/Train/62_celebaMask-HQ-img_10.jpg_imglab.pts",
    sep=";")
txt_np = np.array(txt[2:70])

matrix = []

for line in txt_np:
    for val in line:
        splitted = val.split(' ')
        vals = tuple(int(i) for i in splitted)
        x, y = vals
        matrix.append(vals)
indice = 60
for i in range(0, len(matrix)):
    # x = i[0]
    # y = i[1]
    # print(x ,y)
    img = cv2.circle(image, matrix[i], 4, (255, 0, 255), -1)
    for number, vall in enumerate(matrix, start=0):
        img = cv2.putText(image, str(number), (vall), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # for ind in range(0,len(matrix)):
    if matrix[i] == (535, 750):
        print(matrix[i])
        continue
image.shape
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("frame", image)
key = cv2.waitKey()