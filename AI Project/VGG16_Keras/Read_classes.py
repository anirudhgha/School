
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd


# Step 1: Read the labels, store them in separate variables
df = pd.read_csv("HAM10000_metadata.csv")
train_label = df['dx']
train_id = df['image_id']

