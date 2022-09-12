# %%

from tensorflow import keras
import numpy as np
import augmentation
import normalization
import standardization
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
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from winsound import PlaySound
# %%

runs = 20
array = np.zeros((runs,4))

for i in range(1,runs+1):
    # Define Parameters

    validation_size = 0.25
    epochs = 400
    batch_size = 100
    lr=0.03
    decay = 0

    optimizer = optimizers.Adam(learning_rate=lr)
    myReduce_lr = ReduceLROnPlateau(monitor='val_r2', factor=0.5, patience=15, verbose=2, mode='auto', cooldown=15, min_lr=0)
    tensorboard_logs = keras.callbacks.TensorBoard(log_dir="./logs")

    mycallbacks = [myReduce_lr] # Options: [myReduce_lr],None,[myReduce_lr,tensorboard_logs]

    regularizer = l1(0.01) # Options: l1(0.01),l2(0.01), None

    no1,no2,no3,no4 = 1,2,3,4  # the No. of test output visualization 
    # %%
    # Import training data
    Ytrain = loadmat('D:/Final_project/python_code/data/data-training.mat')
    Xtrain = loadmat('D:/Final_project/python_code/data/trainingImages.mat')
    x_train=Xtrain['Images']
    y_train = Ytrain['Y']
    y_train = y_train.reshape(-1,)

    # import test data
    Ytest = loadmat('D:/Final_project/python_code/data/data-test.mat')
    Xtest = loadmat('D:/Final_project/python_code/data/testImages.mat')
    x_test = Xtest['Images']
    y_test = Ytest['Y']
    y_test = y_test.reshape(-1,)

    x_train=np.transpose(x_train, axes=[3, 0, 1, 2])
    x_train = x_train.reshape(-1,50,50)
    x_test=np.transpose(x_test, axes=[3, 0, 1, 2])
    x_test = x_test.reshape(-1,50,50)   

    # Data standardization 
    y_train = standardization.standardization(y_train)
    y_test = standardization.standardization(y_test)

    # Data normalization
    # y_train = normalization.normalization(y_train)
    # y_test = normalization.normalization(y_test)
    x_train, y_train = shuffle(x_train,y_train)

    x_train, x_validation, y_train, y_validation = train_test_split ( x_train, y_train, shuffle=True, test_size = validation_size )
    # Image augmentation and shuffle the data
    x_train = augmentation.augmentation (x_train)
    y_train = y_train.tolist()
    y_train = (y_train)*4
    y_train  = np.array(y_train)
    x_train, y_train = shuffle(x_train,y_train)

    # Split to training and validation


    # %%
    # Build CNN 1

    model = Sequential()
    model.add(InputLayer (input_shape= (50,50,1)))
    model.add(Conv2D(6, (7, 7),name="Cov1",kernel_regularizer= regularizer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4),name="Pooling1"))
    model.add(Conv2D(1, (1, 1),name="Cov2"))
    model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=optimizer , metrics=[R2.r2])

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
    traintime = t1_stop-t1_start
    print('Training time:', traintime,'seconds')

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