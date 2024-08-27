% Reading data 

clc
clear all
close all
warning off

% loading data after pre processing the data set has 10 features
bcdata = readtable("bc_fs.csv")
% making the training and test data set (90% Training 10% Test)
Train_set_x=bcdata(1:512,2:11);
Train_set_y=bcdata(1:512,["diagnosis"]);
Test_set_x=bcdata(512:end,2:11);
Test_set_y=bcdata(512:end,["diagnosis"]);

%Naive Base Model
% nb_model=fitcnb(Train_set_x,Train_set_y)
% pre=predict(nb_model,Train_set_x)
% 
% inputTable = trainingData;
% predictorNames = {'radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concavePoints_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concavePoints_worst'};
% predictors = inputTable(:, predictorNames);
% response = inputTable.diagnosis;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

distributionNames =  repmat({'Normal'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

classificationNaiveBayes = fitcnb(...
        Train_set_x, ...
        Train_set_y, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [0; 1]);


pre_train=predict(classificationNaiveBayes,Train_set_x)
A=table2array(Train_set_y)
confusionchart(pre_train,A)
matrix = confusionmat(pre_train,A)
matrix=matrix'

%How To Calculate The Model Accuracy
diagonal1 = diag(matrix);
sum_of_rows = sum(matrix,2);
precision = diagonal1 ./ sum_of_rows;
%number of model predict are correct compared to the actual value
overall_precision = mean(precision)
sum_of_columns = sum(matrix,1);
recall = diagonal1 ./ sum_of_columns' ;

%the number of the predictions that the model did correct from the test set
overall_recall = mean(recall)

%precision and recall conmbined to see the overall performance of the model
f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall))

% to check if the model is more precise than accuracy, f1 score will help
% determin wherther we have a good model or not

mdlaccuracy = (diagonal1(1,1)+diagonal1(2,1))/(sum_of_rows(1,1)+sum_of_rows(2,1))

%tunning my mdl
% Got the code from https://uk.mathworks.com/help/stats/fitctree.html
tunedmdlnb = fitcnb(Train_set_x, Train_set_y,'OptimizeHyperparameters','auto')

%%
%% Test set
pre_test=predict(tunedmdlnb,Test_set_x)
A=table2array(Test_set_y)
confusionchart(pre_test,A)
matrix = confusionmat(pre_test,A)
matrix=matrix'

%How To Calculate The Model Accuracy
diagonal1 = diag(matrix);
sum_of_rows = sum(matrix,2);
precision = diagonal1 ./ sum_of_rows;
%number of model predict are correct compared to the actual value
overall_precision = mean(precision)
sum_of_columns = sum(matrix,1);
recall = diagonal1 ./ sum_of_columns' ;

%the number of the predictions that the model did correct from the test set
overall_recall = mean(recall)

%precision and recall conmbined to see the overall performance of the model
f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall))

% to check if the model is more precise than accuracy, f1 score will help
% determin wherther we have a good model or not
mdlaccuracy = (diagonal1(1,1)+diagonal1(2,1))/(sum_of_rows(1,1)+sum_of_rows(2,1))

%%
save('Final_model_1.mat','classificationNaiveBayes')