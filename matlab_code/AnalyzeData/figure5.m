%% Makes figure 5
% Initialize
clear all
format shortg

%% Load Training and Validation Data and do Linear Fits
load('../Data/KantzosCNN.mat')
train = fitlm(predictedY_train,trainY);
validation = fitlm(predictedY_validation,validationY);

%% Load Test Data
load('../Data/data-test.mat')
testY = Y;
testY = (testY-mean(testY))./std(testY);

%% Evaluate Test Data Test
load('../Data/testImages.mat')
predictedY_test = predict(net,Images);
test = fitlm(predictedY_test,testY);

%% Plot Fits
f5 = figure(5);
f5.Color = 'w';
f5.Position = [384         259        1217         546];
%%
subplot(1,3,1)
hold on
train.plot
plot([-4 4],[-4 4],'k','linewidth',3)
axis equal
axis square
axis([-4 4 -4 4])
hold off
xlabel('Predicted Strain Energy')
ylabel('Actual Strain Energy')
title(strcat('Training Data with R^2 = ', num2str(train.Rsquared.Ordinary)))
ax1 = gca;
delete(ax1.Children(2))
delete(ax1.Children(2))
legend('Data','Fit','y=x')
%%
subplot(1,3,2)

hold on
validation.plot
plot([-4 4],[-4 4],'k','linewidth',3)
axis equal
axis square


axis([-4 4 -4 4])
hold off
xlabel('Predicted Strain Energy')
ylabel('Actual Strain Energy')
title(['Validation Data with R^2 = ' num2str(validation.Rsquared.Ordinary)])
ax2 = gca;
delete(ax2.Children(2))
delete(ax2.Children(2))
legend('Data','Fit','y=x')
%%
subplot(1,3,3)

hold on
test.plot
plot([-4 4],[-4 4],'k','linewidth',3)
axis equal
axis square


axis([-4 4 -4 4])
hold off
xlabel('Predicted Strain Energy')
ylabel('Actual Strain Energy')
title(['Test Data with R^2 = ' num2str(test.Rsquared.Ordinary)])
ax3 = gca;
delete(ax3.Children(2))
delete(ax3.Children(2))
legend('Data','Fit','y=x')



set(findall(gcf,'-property','FontWeight'),'FontWeight','bold')
set(findall(gcf,'-property','FontSize'),'FontSize',20)
set(findall(gcf,'-property','LineWidth'),'LineWidth',3)

ax1.Children(3).Marker = '.';
ax1.Children(3).MarkerSize = 8;
ax1.Children(3).MarkerFaceColor = 'b';
ax1.Children(3).LineWidth = 1;

ax2.Children(3).Marker = '.';
ax2.Children(3).MarkerSize = 15;
ax2.Children(3).MarkerFaceColor = 'b';
ax2.Children(3).LineWidth = 1;

ax3.Children(3).Marker = '.';
ax3.Children(3).MarkerSize = 15;
ax3.Children(3).MarkerFaceColor = 'b';
ax3.Children(3).LineWidth = 1;

