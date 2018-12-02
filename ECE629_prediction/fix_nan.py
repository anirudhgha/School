import os
import numpy as np
import pandas as pd


df = pd.DataFrame()
filename = 'full_non_padding.csv'
f = open(filename, 'r')
df = pd.read_csv(filename)
f.close()


print(df)

for i in range(0, len(df[0])):
    for j in range(0, len(df)):
        if df[i, j].empty:
            df[i, j] = df[i, j-1]

