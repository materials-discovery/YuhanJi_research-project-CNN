# %%

from tensorflow import keras
import numpy as np
import augmentation
import normalization
import standardization
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, InputLayer
from keras import optimizers
from keras.regularizers import l1,l2
import R2
from sklearn.metrics import r2_score
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from time import process_time
import ImagePreview_and_plots as ipp
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
print('Packages loaded')

# %%
# Define Parameters

validation_size = 0.25

epochs = 400

batch_size = 48

lr=0.01

decay = 0

optimizer = Adam(learning_rate=lr)

myReduce_lr = ReduceLROnPlateau(monitor='val_r2', factor=0.5, patience=15, verbose=2, mode='auto', cooldown=15, min_lr=0)

tensorboard_logs = keras.callbacks.TensorBoard(log_dir="./logs")

mycallbacks = None
# Options: [myReduce_lr],None,[myReduce_lr,tensorboard_logs]
regularizer = None
# Options: l1(0.01),l2(0.01), None

no1,no2,no3,no4 = 0,1,2,3
# the No. of test images output visualization 

# %%
# Import training data
Ytrain = loadmat('D:/Final_project/python_code/data/data-training.mat')
Xtrain = loadmat('D:/Final_project/python_code/data/trainingImages.mat')
# print(Xtrain.keys())
# print(Ytrain.keys())
x_train=Xtrain['Images']
y_train = Ytrain['Y']
y_train = y_train.reshape(-1,)

# import test data
Ytest = loadmat('D:/Final_project/python_code/data/data-test.mat')
Xtest = loadmat('D:/Final_project/python_code/data/testImages.mat')
# print(Ytest.keys())
# print(Xtest.keys())
x_test = Xtest['Images']
y_test = Ytest['Y']
y_test = y_test.reshape(-1,)

print('Data Loaded',x_train.shape,y_train.shape,x_test.shape,y_test.shape)


x_train=np.transpose(x_train, axes=[3, 0, 1, 2])
x_train = x_train.reshape(-1,50,50)
x_test=np.transpose(x_test, axes=[3, 0, 1, 2])
x_test = x_test.reshape(-1,50,50)   

print('Data reshape completed',x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# Data standardization 
y_train = standardization.standardization(y_train)
y_test = standardization.standardization(y_test)

# Data normalization
# y_train = normalization.normalization(y_train)
# y_test = normalization.normalization(y_test)

print('Data normalization completed')

# Processed image preview
ipp.ImageAugPreview(x_train,no1)
x_train, y_train = shuffle(x_train,y_train)

# Split to training and validation
x_train, x_validation, y_train, y_validation = train_test_split ( x_train, y_train, shuffle=True, test_size = validation_size )
print('Data splitting completed',x_train.shape,y_train.shape,x_validation.shape,y_validation.shape)

# Image augmentation and shuffle the data

x_train = augmentation.augmentation (x_train)
y_train = y_train.tolist()
y_train = (y_train)*4
y_train  = np.array(y_train)
print('add up completed',x_train.shape,y_train.shape)

x_train, y_train = shuffle(x_train,y_train)
print('Image augmentation and data shuffling completed',x_train.shape,y_train.shape)

# %%
# Build CNN

model = Sequential()
model.add(InputLayer (input_shape= (50,50,1)))
model.add(Conv2D(6, (7, 7),name="Cov1",kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4),name="Pooling1"))
model.add(Conv2D(1, (1, 1),name="Cov2"))
model.add(Flatten())
# model.add(Dense(128))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer=Adam(learning_rate=lr) , metrics=[R2.r2])

# %%
# Train CNN
t1_start = process_time() 
history = model.fit (x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          shuffle=True,
          validation_data=(x_validation,y_validation),
          validation_freq=1,
          callbacks=mycallbacks
          )


t1_stop = process_time()

print('Training time:', t1_stop-t1_start,'seconds')

# %%
# Test Plots
y_pred_train = model.predict(x_train,verbose = None)
r2_train = r2_score(y_train, y_pred_train)
print('r2 score for training is', r2_train)
y_pred_train = y_pred_train.reshape(-1,)
y_train= y_train.reshape(-1,)

y_pred_validation = model.predict(x_validation,verbose = None)
r2_validation = r2_score(y_validation, y_pred_validation)
print('r2 score for validation is', r2_validation)
y_pred_validation = y_pred_validation.reshape(-1,)
y_validation = y_validation.reshape(-1,)

y_pred = model.predict(x_test,verbose = None)
r2_test = r2_score(y_test, y_pred)
print('r2 score for test is', r2_test)
y_pred = y_pred.reshape(-1,)
y_test = y_test.reshape(-1,)

ipp.PlotLossR2(history,epochs)

ipp.Plotsfit(y_train,y_pred_train,r2_train,y_validation, y_pred_validation,r2_validation,y_test,y_pred,r2_test)

# ipp.Plotsfit1(y_train,y_pred_train,r2_train,y_validation, y_pred_validation,r2_validation,y_test,y_pred,r2_test)

# ipp.lr_plot(history)

ipp.imageaftercov (model,x_test,no1,no2,no3,no4)

# ipp.Heightlower7 (x_test,no1,no2,no3,no4)

plt.show()


