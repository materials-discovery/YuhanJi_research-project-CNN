%% Extracts Stress Concentration Values from Simulations
numfiles = 512;
siz = 50;
padding = 3;
for n = 1:1:numfiles
n
SVM = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/SVM']);
PH  = h5read(['../Simulations/training-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);

PH2 = PH(:);
SVM = SVM(:);
SVM = sort(SVM(PH2==1));
M(n) = median(SVM);
SVM2 = SVM/median(SVM(:));
SVM2 = SVM2(SVM2>2);
Y(n) = 0.5*sum(SVM2(:).^2)/(20000);



end

save('data-training.mat','Y')
clear all

numfiles = 64;
for n = 1:1:numfiles
n
SVM = h5read(['../Simulations/test-simulations/surf' num2str(n) '/surf.h5'],['/SVM']);
PH  = h5read(['../Simulations/test-simulations/surf' num2str(n) '/surf.h5'],['/PHASE']);

PH2 = PH(:);
SVM = SVM(:);
SVM = sort(SVM(PH2==1));
M(n) = median(SVM);
SVM2 = SVM/median(SVM(:));
SVM2 = SVM2(SVM2>2);
Y(n) = 0.5*sum(SVM2(:).^2)/(20000);

end

save('data-test.mat','Y')
clear all
