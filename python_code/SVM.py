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
import R2
from sklearn.metrics import r2_score
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from time import process_time
import ImagePreview_and_plots as ipp
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from winsound import PlaySound
import pandas as pd
print('Packages loaded')


runs = 10
array = np.zeros((runs,4))


validation_size = 0.25

Ytrain = loadmat('D:/Final_project/python_code/data/data-training.mat')
Xtrain = loadmat('D:/Final_project/python_code/data/trainingImages.mat')
print(Xtrain.keys())
print(Ytrain.keys())
x_train=Xtrain['Images']
y_train = Ytrain['Y']
y_train = y_train.reshape(-1,)

Ytest = loadmat('D:/Final_project/python_code/data/data-test.mat')
Xtest = loadmat('D:/Final_project/python_code/data/testImages.mat')
print(Ytest.keys())
print(Xtest.keys())
x_test = Xtest['Images']
y_test = Ytest['Y']
y_test = y_test.reshape(-1,)

print('Data imported',x_train.shape,y_train.shape,x_test.shape,y_test.shape)


x_train=np.transpose(x_train, axes=[3, 0, 1, 2])
x_train = x_train.reshape(-1,50,50)
x_test=np.transpose(x_test, axes=[3, 0, 1, 2])
x_test = x_test.reshape(-1,50,50)   

print('Data reshape completed',x_train.shape,y_train.shape,x_test.shape,y_test.shape)

y_train = standardization.standardization(y_train)
y_test = standardization.standardization(y_test)

print('Data normalization completed')

x_train = augmentation.augmentation (x_train)
y_train = y_train.tolist()
y_train = (y_train)*4
y_train  = np.array(y_train)
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.
x_train = x_train.reshape(-1,50**2)
x_test = x_test.reshape(-1,50**2) 

x_train, x_validation, y_train, y_validation = train_test_split ( x_train, y_train, shuffle=True, test_size = validation_size )
print('Data splitting completed',x_train.shape,y_train.shape,x_validation.shape,y_validation.shape,)

print('add up completed',x_train.shape,y_train.shape)

x_train, y_train = shuffle(x_train,y_train)
print('Image augmentation and data shuffling completed',x_train.shape,y_train.shape)


for i in range(1,runs+1):
    x_train, y_train = shuffle(x_train,y_train)
    t1_start = process_time() 

    regressor = SVR(kernel = 'rbf')
    regressor.fit(x_train, y_train)

    t1_stop = process_time()
    y_pred = regressor.predict(x_test)
    r2_test = r2_score(y_test, y_pred)
    y_pred_train = regressor.predict(x_train)
    r2_train = r2_score(y_train, y_pred_train)
    y_pred_valid = regressor.predict(x_validation)
    r2_validation = r2_score(y_validation, y_pred_valid)
    print('r2 score for test is', r2_test)
    print('r2 score for train is', r2_train)
    print('r2 score for validation is', r2_validation)
    print('Training time:', t1_stop-t1_start,'seconds')
    traintime =  t1_stop-t1_start
    array[i-1][0]=traintime
    array[i-1][1]=r2_train
    array[i-1][2]=r2_validation
    array[i-1][3]=r2_test
    i = i+1
# auto record experimental data
print(array)
df = pd.DataFrame(array,columns=('training time','R^2 Train','R^2 Validation','R^2 Test'))
print(df)
df.to_csv('experiment_backup.csv')
df.to_csv('experiment.csv')
PlaySound('alarm.wav',flags=10)
