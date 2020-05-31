import pandas as pd
from pprint import pprint

df = pd.read_csv("train.csv")
for row in df.filename:
    print(row)
#  print(row[8])


# a = open('txt.txt', 'w+')
# a.writeline("Hello noob")
# a.close()