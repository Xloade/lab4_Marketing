% Kad pradiniai svoriai nustatomi atsitiktinai kiekviena kart butu vienodi
setdemorandstream(491218382)

% Nuskaitomi ir paruosiami mokymo ir testavimo duomenys
Train = readtable('preprocessed_train.csv');
Test = readtable('preprocessed_test.csv');

Train(:,1) = [];
Test(:,1) = [];

xTrain = Train{:,1:16}';
xTest = Test{:,1:16}';

yTrain = (Train.y)';
yTest = (Test.y)';
%labels = categories(Train.y);

% Apmokinamas modelis
net = patternnet([50,50,50]);
net.trainParam.epochs = 2000;
net.trainParam.lr=0.01;
net.trainParam.max_fail = 500;
%net.trainParam.goal=0.01^2;  %training goal
net.performFcn='mse';
[net,tr] = train(net, xTrain, yTrain);

plotperform(tr)

predictions = net(xTest);
yPred = predictions > 1.5;

% Modelio ivertinimas
yActual = yTest > 1;
accuracy = 100*nnz(yActual == yPred)/length(yPred);
figure
c = plotconfusion(yActual,yPred);
title('Accuracy: '+string(accuracy));
