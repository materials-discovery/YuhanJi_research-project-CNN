%% Makes Figure 8 and 9
clear all

%% Select Surface to visualize
n = 103;

%% Read Data
SVM = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/SVM']);
PH  = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);

%% Get Height Map
z0 = (GetZ2(PH)); % For Stress map

z1 = -(GetZ2(PH)); % For visualization
z1 = z1-round(mean(z1(:)));
z1 = imrotate(fliplr(z1),180);

%% Get Stress Map
PH2 = PH(:);
SVMm = SVM(:);

SVMm = median(sort(SVMm(PH2==1)));
SVM = SVM/median(SVM(:));
SVM = permute(SVM,[3 1 2]);
for i = 1:size(SVM,1)
    for j = 1:size(SVM,2)
        S(i,j) = SVM(i,j,z0(i,j));
        S2(i,j) = max(SVM(i,j,:));
        S3(i,j) = max(SVM(i,j,:));
    end
end

%z1 = imrotate(fliplr(z1),180);
S = imrotate(fliplr(S),180);
S2 = imrotate(fliplr(S2),180);
S3 = imrotate(fliplr(S3),180);
S3(S3<2)=0;

%% Make Plots
f8 = figure(8);
f8.Color = 'w';
f8.Position = [855 259 746 298];
load('data-spatial.mat')
SumBelow = -(SumBelow-mean(SumBelow))./std(SumBelow);
NumberBelow = (NumberBelow-mean(NumberBelow))./std(NumberBelow);

Y = (Y-mean(Y))./std(Y);
FitSum = fitlm(SumBelow,Y);
FitNumber = fitlm(NumberBelow,Y);

subplot(1,2,1)
FitSum.plot
hold on
plot([-4 4],[-4 4],'k','linewidth',3)
axis equal
axis square
axis([-4 4 -4 4])
hold off
xlabel('Sum of points below -7')
ylabel('Actual Strain Energy')
title(strcat('R^2 = ', num2str(FitSum.Rsquared.Ordinary)))
ax1 = gca;
delete(ax1.Children(2))
delete(ax1.Children(2))
legend('Data','Fit','y=x')

ax1.Children(3).Marker = '.';
ax1.Children(3).MarkerSize = 8;
ax1.Children(3).MarkerFaceColor = 'b';
ax1.Children(3).LineWidth = 1;


subplot(1,2,2)
FitNumber.plot
hold on
plot([-4 4],[-4 4],'k','linewidth',3)
axis equal
axis square
axis([-4 4 -4 4])
hold off
xlabel('Number of points below -7')
ylabel('Actual Strain Energy')
title(strcat('R^2 = ', num2str(FitNumber.Rsquared.Ordinary)))
ax1 = gca;
delete(ax1.Children(2))
delete(ax1.Children(2))
legend('Data','Fit','y=x')

ax1.Children(3).Marker = '.';
ax1.Children(3).MarkerSize = 8;
ax1.Children(3).MarkerFaceColor = 'b';
ax1.Children(3).LineWidth = 1;


set(findall(gcf,'-property','FontWeight'),'FontWeight','bold')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','LineWidth'),'LineWidth',3)
%% Plot
f7 = figure(7);
f7.Color = 'w';
f7.Position = [855 142 746 415];
z3 = z1;
z3(z1>-7)=0;

subplot(2,2,1)
cnnplot(z1)
colorbar
ax = gca;
ax.Colormap = (parula(256));
caxis([-15 15])
title('Height Map')

subplot(2,2,2)
cnnplot(S)
colorbar
ax = gca;
ax.Colormap = (jet(256));
caxis([0 2])
title('Surface Stress')

subplot(2,2,3)
cnnplot(z3)
colorbar
ax = gca;
ax.Colormap = (parula(256));
caxis([-15 15])
title('Height Map (below -7)')

subplot(2,2,4)
cnnplot(S3)
colorbar
ax = gca;
ax.Colormap = (jet(256));
caxis([0 2])
title('Surface Stress Concentrations')

set(findall(gcf,'-property','FontWeight'),'FontWeight','bold')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','LineWidth'),'LineWidth',3)

