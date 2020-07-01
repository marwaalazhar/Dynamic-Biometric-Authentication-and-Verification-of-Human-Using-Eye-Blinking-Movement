import datetime
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
plt.style.use('classic')
#Load Dataset
x_train = np.load('dataset/x_train_last.npy').astype(np.float32)
y_train = np.load('dataset/y_train_last.npy').astype(np.float32)
x_val = np.load('dataset/x_val_last.npy').astype(np.float32)
y_val = np.load('dataset/y_val_last.npy').astype(np.float32)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

#Preview
##plt.subplot(2, 1, 1)
##plt.title(str(y_train[0]))
##plt.imshow(x_train[0].reshape((24, 24)), cmap='gray')
##plt.subplot(2, 1, 2)
##plt.title(str(y_val[4]))
##plt.imshow(x_val[4].reshape((24, 24)), cmap='gray')

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x=x_train, y=y_train,
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow(
    x=x_val, y=y_val,
    batch_size=32,
    shuffle=False
)

#Build Model
inputs = Input(shape=(24, 24, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

#Train

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

history = model.fit_generator(
    train_generator, epochs=70, validation_data=val_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ]
)

##history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



