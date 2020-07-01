import datetime
import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
from keras.models import load_model

x_val = np.load('dataset/x_val_last.npy').astype(np.float32)
y_val = np.load('dataset/y_val_last.npy').astype(np.float32)
x_train = np.load('dataset/x_train_last.npy').astype(np.float32)
y_train = np.load('dataset/y_train_last.npy').astype(np.float32)

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model = load_model("models/2019_08_01_12_11_29.h5".format(start_time))

y_pred = model.predict(x_val/255.)
y_pred_logical = (y_pred > 0.5).astype(np.int)

print ('test acc: %s' % accuracy_score(y_val, y_pred_logical))

print(confusion_matrix(y_val, y_pred_logical))
#cm = confusion_matrix(y_val, y_pred_logical)
#sns.heatmap(cm, annot=True)
#ax = sns.distplot(y_pred, kde=False)



# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_val, y_pred_logical)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_val, y_pred_logical)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_val, y_pred_logical)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_val, y_pred_logical)
print('F1 score: %f' % f1)
