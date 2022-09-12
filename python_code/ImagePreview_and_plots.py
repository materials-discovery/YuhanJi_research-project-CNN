import augmentation
import matplotlib.pyplot as plt
from matplotlib.pyplot import gray, title

import numpy as np
import keras.backend as K
def ImageAugPreview (data,no1):
    original = data
    mirror = augmentation.mirror(original)
    rot180 = augmentation.rot180(original)
    mirror_rot180 = augmentation.mirror_rot180(original)
    rows = 2
    columns = 2
    fig, axes = plt.subplots(rows, columns, figsize=(9,8),constrained_layout=True)
    im1 = axes[0,0].imshow(original[no1], cmap='viridis')
    axes[0,0].set_title('Original')
    im2 = axes[0,1].imshow(mirror[no1], cmap='viridis')
    axes[0,1].set_title('Mirrored')
    im3 = axes[1,0].imshow(rot180[no1], cmap='viridis')
    axes[1,0].set_title('Rotate 180°')
    im4 = axes[1,1].imshow(mirror_rot180[no1], cmap='viridis')
    axes[1,1].set_title('Mirrored and Rotated 180°')
    cbar = fig.colorbar(im1, ax=axes[:, 0], shrink=0.8, label = 'Height(px)',location='left')


def Plotsfit (y_train,y_pred_train,r2_train,y_validation, y_pred_validation,r2_validation,y_test,y_pred,r2_test):
    # use this when standardized
    
    s = 2
    fig, ax1 = plt.subplots(figsize=(4,4))
    ax1.plot(np.unique(y_train), np.poly1d(np.polyfit(y_train, y_pred_train, 1))(np.unique(y_train)),color = 'red',label = 'fit')
    ax1.plot(y_train, y_train, color = 'black', label = 'x=y')
    ax1.scatter(y_pred_train,y_train,s = s,label = 'data')
    ax1.text(1.8, -1.5, 'R^2 = %0.2f' % r2_train)
    ax1.axis([-4, 4, -4, 4])    
    ax1.set_xlabel('Predicted Strain Energy')
    ax1.set_ylabel('Actual Strain Energy')
    ax1.set_title('Training fit')
    ax1.legend(loc='lower right')

    fig, ax2 = plt.subplots(figsize=(4,4))
    ax2.plot(np.unique(y_validation), np.poly1d(np.polyfit(y_validation, y_pred_validation, 1))(np.unique(y_validation)),color = 'red',label = 'fit')
    ax2.plot(y_validation, y_validation, color = 'black', label = 'x=y')
    ax2.scatter(y_pred_validation,y_validation,s = s,label = 'data')
    ax2.text(1.8, -1.5, 'R^2 = %0.2f' % r2_validation)
    ax2.axis([-4, 4, -4, 4])
    ax2.set_xlabel('Predicted Strain Energy')
    ax2.set_ylabel('Actual Strain Energy')
    ax2.set_title('Scores by group and gender')
    ax2.set_title('Validation fit')
    ax2.legend(loc='lower right')


    fig, ax3 = plt.subplots(figsize=(4,4))
    ax3.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),color = 'red',label = 'fit')
    ax3.plot(y_test, y_test, color = 'black', label = 'x=y')
    ax3.scatter(y_pred,y_test,s = s,label = 'data')
    ax3.text(1.8, -1.5, 'R^2 = %0.2f' % r2_test)
    ax3.axis([-4, 4, -4, 4])
    ax3.set_xlabel('Predicted Strain Energy')
    ax3.set_ylabel('Actual Strain Energy')
    ax3.set_title('Scores by group and gender')
    ax3.set_title('Test fit')
    ax3.legend(loc='lower right')



def PlotLossR2 (history,epochs):

    markersize = 1.5
    linewidth = 0.7

    fig, ax4 = plt.subplots()
    
    ax4.plot(history.history['loss'],'o',ls='-' ,ms=markersize,label='loss',linewidth=linewidth)
    ax4.plot(history.history['val_loss'],'o', ls='-',ms=markersize,label = 'val_loss',linewidth=linewidth)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('loss')
    ax4.axis([0,epochs,0,5])
    ax4.legend(loc='upper right')
    ax4.set_title('Training loss')
    ax4.grid()
    
    fig, ax5 = plt.subplots()
    
    ax5.plot(history.history['r2'],'o', ls='-',ms=2,label='r2',linewidth=linewidth)
    ax5.plot(history.history['val_r2'],'o',ls='-', ms=2,label = 'val_r2',linewidth=linewidth)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('r2')
    ax5.axis([0,epochs,-1,1])
    ax5.legend(loc='lower right')
    ax5.set_title('r2')
    ax5.grid()
    
def Plotsfit1 (y_train,y_pred_train,r2_train,y_validation, y_pred_validation,r2_validation,y_test,y_pred,r2_test):
    # use this when normalized
    s = 2
    fig, ax1 = plt.subplots(figsize=(4,4))
    ax1.plot(np.unique(y_train), np.poly1d(np.polyfit(y_train, y_pred_train, 1))(np.unique(y_train)),color = 'red',label = 'fit')
    ax1.plot(y_train, y_train, color = 'black', label = 'x=y')
    ax1.scatter(y_pred_train,y_train,s = s,label = 'data')
    ax1.text(0.7, 0.3, 'R^2 = %0.2f' % r2_train)
    ax1.axis([0, 1, 0, 1])    
    ax1.set_xlabel('Predicted Strain Energy')
    ax1.set_ylabel('Actual Strain Energy')
    ax1.set_title('Training fit')
    ax1.legend(loc='lower right')

    fig, ax2 = plt.subplots(figsize=(4,4))
    ax2.plot(np.unique(y_validation), np.poly1d(np.polyfit(y_validation, y_pred_validation, 1))(np.unique(y_validation)),color = 'red',label = 'fit')
    ax2.plot(y_validation, y_validation, color = 'black', label = 'x=y')
    ax2.scatter(y_pred_validation,y_validation,s = s,label = 'data')
    ax2.text(0.7, 0.3, 'R^2 = %0.2f' % r2_validation)
    ax2.axis([0, 1, 0, 1])
    ax2.set_xlabel('Predicted Strain Energy')
    ax2.set_ylabel('Actual Strain Energy')
    ax2.set_title('Scores by group and gender')
    ax2.set_title('Validation fit')
    ax2.legend(loc='lower right')


    fig, ax3 = plt.subplots(figsize=(4,4))
    ax3.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)),color = 'red',label = 'fit')
    ax3.plot(y_test, y_test, color = 'black', label = 'x=y')
    ax3.scatter(y_pred,y_test,s = s,label = 'data')
    ax3.text(0.7, 0.3, 'R^2 = %0.2f' % r2_test)
    ax3.axis([0, 1, 0, 1])
    ax3.set_xlabel('Predicted Strain Energy')
    ax3.set_ylabel('Actual Strain Energy')
    ax3.set_title('Scores by group and gender')
    ax3.set_title('Test fit')
    ax3.legend(loc='lower right')




def lr_plot(history):

    fig, ax4 = plt.subplots()
    ax4.plot(history.history['lr'],ls='-' ,label='learning rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('learning rate')
    ax4.set_title('learning rate')
    ax4.grid()   




def imageaftercov(model,data,no1,no2,no3,no4):
    func = K.function([model.get_layer('Cov1').input], model.get_layer('Cov2').output)
    conv_output = func([data])  # numpy array
    rows = 2 
    columns = 4 

    fig, axes = plt.subplots(rows, columns, figsize=(15,7),constrained_layout=True)
    im1 = axes[0,0].imshow(data[no1], cmap='viridis')
    axes[0,0].set_title('Origin_1')
    im2 = axes[0,1].imshow(data[no2], cmap='viridis')
    axes[0,1].set_title('Origin_2')
    im3 = axes[0,2].imshow(data[no3], cmap='viridis')
    axes[0,2].set_title('Origin_3')
    im4 = axes[0,3].imshow(data[no4], cmap='viridis')
    axes[0,3].set_title('Origin_4')
    im5 = axes[1,0].imshow(conv_output[no1], cmap='gray')
    axes[1,0].set_title('after_conv_1')
    im6 = axes[1,1].imshow(conv_output[no2], cmap='gray')
    axes[1,1].set_title('after_conv_2')
    im7 = axes[1,2].imshow(conv_output[no3], cmap='gray')
    axes[1,2].set_title('after_conv_3')
    im8 = axes[1,3].imshow(conv_output[no4], cmap='gray')    
    axes[1,3].set_title('after_conv_4')  
    cbar1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.8, label = 'Height(px)',location='left')
    cbar2 = fig.colorbar(im5, ax=axes[1, 0], shrink=0.8, label = 'Activation',location='left')

def Preview(data,no1,no2,no3,no4):
    rows = 1
    columns = 4

    fig, axes = plt.subplots(rows, columns, figsize=(14,3),constrained_layout=True)
    im1 = axes[0].imshow(data[no1], cmap='viridis')
    axes[0].set_title('1')
    im2 = axes[1].imshow(data[no2], cmap='viridis')
    axes[1].set_title('2')
    im3 = axes[2].imshow(data[no3], cmap='viridis')
    axes[2].set_title('3')
    im4 = axes[3].imshow(data[no4], cmap='viridis')
    axes[3].set_title('4')

    cbar = fig.colorbar(im1,ax=axes[0] , shrink=0.8, label = 'Height(px)',location='left')

def Heightlower7 (data_original,no1,no2,no3,no4):
    rows = 1
    columns = 4
    data = data_original
    data[data > -7 ] = -100
    fig, axes = plt.subplots(rows, columns, figsize=(14,3),constrained_layout=True)
    im1 = axes[0].imshow(data[no1], cmap='viridis')
    axes[0].set_title('Lower_than_-7_1')
    im2 = axes[1].imshow(data[no2], cmap='viridis')
    axes[1].set_title('Lower_than_-7_2')
    im3 = axes[2].imshow(data[no3], cmap='viridis')
    axes[2].set_title('Lower_than_-7_3')
    im4 = axes[3].imshow(data[no4], cmap='viridis')
    axes[3].set_title('Lower_than_-7_4')
