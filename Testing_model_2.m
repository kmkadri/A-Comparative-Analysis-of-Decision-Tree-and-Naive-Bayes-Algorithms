load ('Final_model_2.mat')

%% accuracy for test set
pre_test=predict(tunedmdl,Test_set_x)
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