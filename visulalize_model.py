import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.utils import plot_model

model = load_model('models/2019_08_01_12_11_29.h5')
model.summary()
plot_model(model, to_file='Model_architecture2.png')





