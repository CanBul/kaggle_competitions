import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model

X = pd.read_csv('train.csv')
print(X.shape)

Y = X['label']
X.drop('label', axis=1, inplace=True)

X= X.values.reshape(-1,28,28,1)
X= X/255.0

Y = to_categorical(Y, 10)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(x_train)

model = Sequential()

model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

earlyStopping = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='max')
mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=0.00001)
batch_size =128
print('Training has started...')
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), verbose=2,
                              epochs = 100, validation_data = (x_test,y_test), steps_per_epoch=x_train.shape[0] // batch_size, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
