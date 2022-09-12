%% Makes figure 6
%% Initialize
clear all
load KantzosCNN.mat
app.UIFigure.HandleVisiblity = 'on'; 
colormapeditor

%%
n = 3; % Surface to look at 

%% Load Data 
SVM = h5read(['../Simulations/test-simulations/surf' num2str(n) '/surf.h5'],['/SVM']);
PH  = h5read(['../Simulations/test-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);

%% Get Height Map(s)
%Extract height map for CNN
z = (GetZ(PH)); 
img = padarray(imresize(imrotate(fliplr(z),180),[siz-2*padding siz-2*padding],'bicubic'),[padding padding],'circular');
act6 = activations(net,img,6);

% Extract height map for Surface Stresses
z0 = (GetZ2(PH)); 

% Extract height map for  visualization
z1 = -(GetZ2(PH)); 
z1 = z1-round(mean(z1(:)));
z1 = imrotate(fliplr(z1),180);


%% Generate Stress Map
PH2 = PH(:);
SVMm = SVM(:);

SVMm = median(sort(SVMm(PH2==1)));
SVM = permute(SVM,[3 1 2]);
for i = 1:size(SVM,1)
    for j = 1:size(SVM,2)
        S(i,j) = SVM(i,j,z0(i,j));
    end
end
S = imrotate(fliplr(S),180);

%% Generate Stress Concentration Map

for i = 1:size(SVM,1)
    for j = 1:size(SVM,2)
        S2(i,j) = max(SVM(i,j,:));
    end
end

S2(S2<2*SVMm) = 0;
S2 = imrotate(fliplr(S2),180);


%% Make plots
f6 = figure(6);
f6.Color = 'w';
f6.Position = [67         436        1534         369];

subplot(1,5,1)
cnnplot(z1)
caxis([-15 15])
c1 = colorbar('location', 'southoutside');
c1.Label.String = 'Height (px)';
ax = gca;
ax.Colormap =parula(256);
title('Original Height Map')

subplot(1,5,2)
cnnplot(img)
caxis([-15 15])
c2 = colorbar('location', 'southoutside');
c2.Label.String = 'Height (px)';
ax = gca;
ax.Colormap = (parula(256));
title('Scaled and Padded')

subplot(1,5,3)
cnnplot(S)
caxis([0 round(2*SVMm)])
c3 = colorbar('location', 'southoutside');
c3.Label.String = 'Stress (MPa)';
ax = gca;
ax.Colormap = (jet(256));
title('Stress Map')

subplot(1,5,4)
cnnplot(S2)
caxis([1.2*SVMm max(S2(:))])
c4 = colorbar('location', 'southoutside');
c4.Label.String = 'Stress (MPa)';
ax = gca;
ax.Colormap = (hot(256));
title('Stress Concentrations (>2\sigma_0)')

subplot(1,5,5)
cnnplot(-act6)
c5 = colorbar('location', 'southoutside');
c5.Label.String = 'Activation';
ax = gca;
ax.Colormap = (gray(256));
title('Output from Conv2')

set(findall(gcf,'-property','FontWeight'),'FontWeight','bold')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','LineWidth'),'LineWidth',3)

app.UIFigure.HandleVisiblity = 'off'; 