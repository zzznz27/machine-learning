import pandas as pd
from pprint import pprint
import shutil, os


df = pd.read_csv("train.csv")
for row in df.filename:
    try:
        shutil.move("images/"+row, 'train')
    except:
        continue
#  print(row[8])


# a = open('txt.txt', 'w+')
# a.writeline("Hello noob")
# a.close()