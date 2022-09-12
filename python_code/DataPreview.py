from tensorflow import keras
import numpy as np
import augmentation
import normalization
import standardization
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import ImagePreview_and_plots as ipp
print('Packages loaded')

no1,no2,no3,no4 = 0,1,2,3

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

# Processed image preview
# ipp.ImageAugPreview(x_train,no1)


print('Data reshape completed',x_train.shape,y_train.shape,x_test.shape,y_test.shape)
ipp.Preview(x_train,no1,no2,no3,no4)
# Data standardization 
y_train = standardization.standardization(y_train)
y_test = standardization.standardization(y_test)

# Data normalization
# y_train = normalization.normalization(y_train)
# y_test = normalization.normalization(y_test)

print('Data normalization completed')

# Processed image preview
# ipp.ImageAugPreview(x_train,no1)

# Image augmentation and shuffle the data

x_train = augmentation.augmentation (x_train)
y_train = y_train.tolist()
y_train = (y_train)*4
y_train  = np.array(y_train)
print('add up completed',x_train.shape,y_train.shape)

x_train, y_train = shuffle(x_train,y_train)
print('Image augmentation and data shuffling completed',x_train.shape,y_train.shape)

plt.show()