%% Load Training Data
siz = 50;
padding = 3;
disp('Loading Images . . .')
i = 1;
num_images = 512;
for n = 1:num_images
    disp(['Image ' num2str(n) ' of ' num2str(num_images) '.'])
    PH = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);
    z = (GetZ(PH));
    Images(:,:,1,n) = padarray(imresize(imrotate(fliplr(z),180),[siz-2*padding siz-2*padding],'bicubic'),[padding padding],'circular');
end
save('trainingImages.mat','Images','siz','padding','num_images')

%%
clear all
siz = 50;
padding = 3;
disp('Loading Images . . .')
i = 1;
num_images = 64;
for n = 1:num_images
    disp(['Image ' num2str(n) ' of ' num2str(num_images) '.'])
    PH = h5read(['../Simulations/test-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);
    z = (GetZ(PH));
    Images(:,:,1,n) = padarray(imresize(imrotate(fliplr(z),180),[siz-2*padding siz-2*padding],'bicubic'),[padding padding],'circular');
end
save('testImages.mat','Images','siz','padding','num_images')