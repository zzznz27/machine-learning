import pandas as pd
import shutil, os
import cv2
from tqdm import tqdm

import csv

f = open('train.csv')
csv_f = csv.reader(f)

for row in tqdm(csv_f):
    try:
        path = 'images/train/'+row[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        crop_img =img[int(row[2]):int(row[4]), int(row[1]):int(row[3])] 
        cv2.imwrite('images/testimg/'+row[0],crop_img)

    except Exception as e:
        print(str(e))
        pass
        
