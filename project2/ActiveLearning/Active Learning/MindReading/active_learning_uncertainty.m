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
    
    newProb_table=zeros(size(unlabeledMatrix, 1), num_classes);
    for i=1:size(unlabeledMatrix, 1)
        newProb_table(i, :)=test_LR_Classifier(unlabeledMatrix(i, :), weight, num_classes);
    end;
        
    [row, col]=size(unlabeledMatrix);
    sample=zeros(k, col);
    sample_labels=zeros(k, 1);
    for j=1:row
        e=0;
        for t=1:num_classes
            e=e+newProb_table(j, t)*log(newProb_table(j, t));
        end;
        entropy_table(j, 1)=-e;
        entropy_table(j, 2)=j;
    end;
    entropy_table=sortrows(entropy_table, -1);
    sample_indices=entropy_table(1:k, 2);
    for j=1:k
        sample(j, :)=unlabeledMatrix(sample_indices(j), :);
        sample_labels(j, :)=unlabeledLabels(sample_indices(j), :);
    end;  
    unlabeledMatrix(sample_indices, :)=[];
    unlabeledLabels(sample_indices, :)=[];
    trainingMatrix=[trainingMatrix; sample];
    trainingLabels=[trainingLabels; sample_labels];
end;
plot(accuracy_table);
