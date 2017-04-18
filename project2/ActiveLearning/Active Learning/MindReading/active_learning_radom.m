clear 
clc
close all

load('trainingMatrix_MindReading1.mat');
load('trainingLabels_MindReading_1.mat');
load('testingMatrix_MindReading1.mat');
load('testingLabels_MindReading1.mat');
load('unlabeledMatrix_MindReading1.mat');
load('testingLabels_MindReading1.mat')
load('unlabeledMatrix_MindReading1.mat')
load('unlabeledLabels_MindReading_1.mat')

N = 50;
k = 10;
num_classes=size(unique(trainingLabels), 1);
accuracy_table=zeros(N, 1);
for iter=1:N
    [weight]=train_LR_Classifier(trainingMatrix, trainingLabels, num_classes);
    prob_table=zeros(size(testingMatrix, 1), num_classes);
    for i=1:size(testingMatrix, 1)
        prob_table(i, :)=test_LR_Classifier(testingMatrix(i, :), weight, num_classes);
    end;
    predicted_labels=zeros(size(testingMatrix, 1), 1);
    for i=1:size(testingMatrix, 1)
        label=1;
        max=prob_table(1);
        for j=2:num_classes
            if prob_table(i, j)>max
                label=j;
                max=prob_table(i, j);
            end;
            predicted_labels(i)=label;
        end;
    end; 
    count=0;
    for i=1:size(testingMatrix, 1)
        if predicted_labels(i)==testingLabels(i)
            count=count+1;
        end;
    end;
    acc = count/size(testingMatrix, 1);
    accuracy_table(iter)=acc;
    
    [row, col]=size(unlabeledMatrix);
    sample=zeros(k, col);
    sample_labels=zeros(k, 1);
    for j=1:k
        idx=randi(row-j+1);
        sample(j, :) = unlabeledMatrix(idx, :);
        sample_labels(j, :)=unlabeledLabels(idx, :);
        unlabeledMatrix(idx, :)=[];
        unlabeledLabels(idx, :)=[];
    end;  
    trainingMatrix=[trainingMatrix; sample];
    trainingLabels=[trainingLabels; sample_labels];
end;
plot(accuracy_table);
