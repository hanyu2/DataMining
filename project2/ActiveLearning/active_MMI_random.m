clear 
clc
close all
clear 
clc
close all

numfiles = 3;
accuracies = zeros(50,numfiles);
avg = zeros(50,1);
avg = sum(accuracies, 1);
avg = avg./50;
for idxfiles = 1:numfiles
    trainMatrix_MR = sprintf('trainingMatrix_%d.mat', idxfiles);
    trainLabels_MR = sprintf('trainingLabels_%d.mat',idxfiles);
    testMatrix_MR = sprintf('testingMatrix_%d.mat',idxfiles);
    testLabels_MR = sprintf('testingLabels_%d.mat',idxfiles);
    unlabeledMatrix_MR = sprintf('unlabeledMatrix_%d.mat',idxfiles);
    unlabeledLabels_MR = sprintf('unlabeledLabels_%d.mat',idxfiles);
    load(trainMatrix_MR);
    load(trainLabels_MR);
    load(testMatrix_MR);
    load(testLabels_MR);
    load(unlabeledLabels_MR);
    load(unlabeledMatrix_MR);
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
            max=prob_table(i,1);
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
        accuracies(iter, idxfiles) = acc;


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
    accuracies(:,idxfiles) = accuracy_table;

end
avg = zeros(50,1);
x = sum(accuracies, 2);
avg = x;
avg = avg./3;
figure(2);
title('MMI with RANDOM Sampling')
hold on;
plot(accuracies(:,1),'-k');
plot(accuracies(:,2),'-m');
plot(accuracies(:,3),'-c');
plot(avg,'--r'); 
legend('run1', 'run2', 'run3', 'avg', 'Location','SouthEast');
grid on
hold off;
