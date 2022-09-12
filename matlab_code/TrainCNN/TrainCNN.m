%% Train a Convolutional Neural Network for Regression
%%
clear all
close all
clc
%% Load and shuffle training images
load('../Data/trainingImages.mat')
num_images = 512;
idx_shuf = randsample(num_images,num_images);
Images = Images(:,:,:,idx_shuf);

%% shuffle and normalize training data
load('../Data/data-training.mat')
Y = Y-mean(Y);
Y = Y./std(Y);
Y = Y'; 
Y = Y(idx_shuf);
totalY = Y;

%% Seperate into validation and train, and apply image segmentation

% Training
train_percent = 7.5/10;
idx = randperm(num_images,round(train_percent*num_images));

trainImages1   = Images(:,:,:,idx);
trainImages2   = imrotate(trainImages1,180);
trainImages3   = fliplr(trainImages1);
trainImages4   = imrotate(trainImages3,180);

trainImages = cat(4,trainImages1, ...
                    trainImages2, ...
                    trainImages3, ...
                    trainImages4);
               
Images(:,:,:,idx) = [];

% Validation
validationImages = Images;
trainY = repmat(totalY(idx),[4,1]);
totalY(idx) = [];
validationY = totalY;

%% Reshuffle training data
num_train = size(trainImages,4);
idx = randperm(num_train,num_train);
trainImages   = trainImages(:,:,:,idx);
trainY = trainY(idx);

%% Define Network
layers = [
imageInputLayer([siz siz 1])

convolution2dLayer([7 7],6)
batchNormalizationLayer
reluLayer
maxPooling2dLayer(4,'Stride',4)

convolution2dLayer([1 1],1)
fullyConnectedLayer(1)
regressionLayer
];


%% Network Taining Options
options  =  trainingOptions('adam',...
                            'MiniBatchSize',48, ...
                            'Shuffle','every-epoch',...
                            'MaxEpochs',100, ...
                            'InitialLearnRate',0.01,...
                            'LearnRateSchedule','piecewise',...
                            'LearnRateDropPeriod',100, ...
                            'LearnRateDropFactor',0.5,...
                            'ValidationData',{validationImages,validationY},...
                            'ValidationFrequency',2,...
                            'ValidationPatience',Inf,...
                            'Plots','training-progress',...
                            'Verbose',true);

%% Train Network
net = trainNetwork(trainImages,trainY,layers,options);
    
%% When Training is Complete make a final prediction images
predictedY_validation = predict(net,validationImages);
predictedY_train = predict(net,trainImages);

%% Save Data 
save('ExampleCNN.mat','net','siz','padding','predictedY_train','predictedY_validation','validationY','trainY')

%% Plot Fits
plotfitswithtestdata
