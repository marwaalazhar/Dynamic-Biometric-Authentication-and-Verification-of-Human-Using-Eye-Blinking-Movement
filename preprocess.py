from helpers import *
import matplotlib.pyplot as plt
import os, glob, cv2, random
import seaborn as sns
import pandas as pd

#Preview
base_path = 'dataset'

X, y = read_csv(os.path.join(base_path, 'dataset/data_zju.csv'))

print(X.shape, y.shape)


# When I saw the left eye
plt.figure(figsize=(12, 10))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(X[i].reshape((36, 48)), cmap='gray')

sns.distplot(y, kde=False)

#Preprocessing
n_total = len(X)
X_result = np.empty((n_total, 36, 48, 1))

for i, x in enumerate(X):
    img = x.reshape((36, 48, 1))
    
    X_result[i] = img
    
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.2)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

np.save('dataset/x_train_last.npy', x_train)
np.save('dataset/y_train_last.npy', y_train)
np.save('dataset/x_val_last.npy', x_val)
np.save('dataset/y_val_last.npy', y_val)

plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((36, 48)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_val[4]))
plt.imshow(x_val[4].reshape((36, 48)), cmap='gray')

sns.distplot(y_train, kde=False)
sns.distplot(y_val, kde=False)

