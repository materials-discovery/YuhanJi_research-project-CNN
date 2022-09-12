clear all
close all
clc
load KantzosCNN.mat

nums = [103 500 381];
for p = 1:length(nums)
    n = nums(p);
SVM = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/SVM']);
PH  = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);
z = (GetZ(PH));
z0 = (GetZ2(PH));
z1 = -(GetZ2(PH));
z1 = z1-round(mean(z1(:)));
z1 = imrotate(fliplr(z1),180);
img = padarray(imresize(imrotate(fliplr(z),180),[siz-2*padding siz-2*padding],'bicubic'),[padding padding],'circular');

act6 = activations(net,img,6);
last = act6.*net.Layers(7).Weights;

SVM = SVM/median(SVM(:));

PH2 = PH(:);
SVMm = SVM(:);

SVMm = 1;%median(sort(SVMm(PH2==1)));
SVM = permute(SVM,[3 1 2]);
for i = 1:size(SVM,1)
    for j = 1:size(SVM,2)
        S(i,j) = SVM(i,j,z0(i,j));
    end
end

for i = 1:size(SVM,1)
    for j = 1:size(SVM,2)
        S2(i,j) = max(SVM(i,j,:));
    end
end

S2(S2<2*SVMm) = 0;

S = imrotate(fliplr(S),180);

S2 = imrotate(fliplr(S2),180);

f9 = figure(9);
f9.Color = 'w';
f9.Position = [672 -517 929 1255];

r = 1+3*(p-1);

subplot(3,3,r)
cnnplot(z1)
caxis([-15 15])
c1 = colorbar('location', 'southoutside');
c1.Label.String = 'Height (px)';
ax = gca;
ax.Colormap =parula(256);
title('Original Height Map')

subplot(3,3,r+1)
cnnplot(S2)
caxis([1 3])
c4 = colorbar('location', 'southoutside');
c4.Label.String = 'Normalized Stress';
ax = gca;
ax.Colormap = (hot(256));
title('Stress Concentrations (>2\sigma_0)')

subplot(3,3,r+2)
cnnplot(-act6)
caxis([0 0.5])
c5 = colorbar('location', 'southoutside');
c5.Label.String = 'Activation';
ax = gca;
ax.Colormap = (gray(256));
title('Output from Conv2')
end

set(findall(gcf,'-property','FontWeight'),'FontWeight','bold')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','LineWidth'),'LineWidth',3)