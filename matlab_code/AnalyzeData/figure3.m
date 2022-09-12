%% Makes Figure 3
clear all
%% Select Surface to look at
rid = 103; 

%% Prepare surface
siz = 50;
padding = 3;
i = 1;
num_images = 512;

PH  = h5read(['../Simulations/training-simulations/surf' num2str(rid) '/surf.h5'],['/PHASE']);

z = (GetZ(PH));
imgg = padarray(imresize(imrotate(fliplr(z),180),[siz-2*padding siz-2*padding],'bicubic'),[padding padding],'circular');

%% Plot surfaces
f3 = figure(3);
subplot(2,2,1)
imagesc(imgg);
axis equal
axis off
colormap parula
colorbar
caxis([-15 15])
title('Original Image')
caxis([-15 15])

subplot(2,2,2)
imagesc(imrotate(imgg,180));
axis equal
axis off
colormap parula
colorbar
caxis([-15 15])
title('Rot 180')
caxis([-15 15])

subplot(2,2,3)
imagesc(fliplr(imgg));
axis equal
axis off
colormap parula
colorbar
caxis([-15 15])
title('Mirrored')
caxis([-15 15])

subplot(2,2,4)
imagesc(fliplr(imrotate(imgg,180)));
axis equal
axis off
colormap parula
colorbar
caxis([-15 15])
title('Mirrored, Rot 180')
caxis([-15 15])

f3.Color = 'w';
set(findall(gcf,'-property','FontSize'),'FontSize',20)