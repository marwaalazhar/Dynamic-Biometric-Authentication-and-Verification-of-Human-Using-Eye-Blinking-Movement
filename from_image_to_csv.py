from PIL import Image
import os, sys
from helpers import *
import matplotlib.pyplot as plt
import os, glob, cv2, random
import seaborn as sns
import pandas as pd

base_path = '/implemntation/my-project/dataset'

X, y = read_csv(os.path.join(base_path, 'data_zju.csv'))

print(X.shape, y.shape)

plt.figure(figsize=(12, 10))
for i in range(10):
 plt.subplot(10, 5, i+1)
 cv2.imshow('', X[i].reshape((36, 48)))

