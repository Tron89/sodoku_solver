import pandas as pd
import cv2
import numpy as np
import os
img_list = []
i = 0
none = []
for actual_path, directories, files in os.walk(os.getcwd()):
    if directories == none:
        for file in files:
            image_file = os.path.join(actual_path, file)
            imagen = cv2.imread(image_file)
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
            thresh = 255 - thresh
            thresh = thresh.reshape(1, -1)
            print(thresh)
            print(len(thresh[0]))
            img_num = np.append(thresh, i)
            img_list.append(img_num)
        i += 1

img_vec_2d = np.vstack(img_list)
df = pd.DataFrame(img_vec_2d)

df.to_csv("dataset.csv", index=False)
input("Mision completed")