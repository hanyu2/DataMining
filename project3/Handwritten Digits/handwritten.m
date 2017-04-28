load('X_test')
load('X_train')
load('y_test')
load('y_train')

num_classes=size(unique(y_train), 1);
[sample_size, attribute_size] = size(X_test);
k = 7;
predict_table=zeros(sample_size, num_classes);

[train_rows, train_cols] = size(X_train);
% training data set
% training data set
model_knn = fitcknn(X_train, y_train, 'NumNeighbors', k);
predict_knn = predict(model_knn, X_test);

% percentage of matching
per_knn = sum(predict_knn == y_test) / size(y_test, 1) * 100;
% fprintf('The accuracy using k-nearest neighbor with k = 5 is %.2f%%. \n', per_knn * 100);
fprintf('If, k = 5, KNN Precsion is %.2f%%. \n', per_knn);

% Support Vector Machine method
% Reference: https://www.mathworks.com/help/stats/fitcsvm.html
kernel = 2;
labels = unique(y_train);
num_labels = numel(labels);

% model_svm = fitcsvm(x_train, y_train, 'KernelFunction', 'polynomial', 'Standardize', true, 'ClassNames', {'1'});
models_svm = cell(num_labels, 1);
y_train_new=zeros(train_rows, num_labels);
for i=1:train_rows
    for j=1:num_labels
        if y_train(i,1) == j
            y_train_new(i, j)=1;
        end;
    end;
end;

for i=1:num_labels
    SVMModel=fitcsvm(X_train, y_train_new(:,i),'KernelFunction', 'polynomial', 'PolynomialOrder',kernel, 'NumPrint', 10000);
    models_svm{i, 1}=SVMModel;
end;

Scores = zeros(sample_size, num_labels);

% create matrix to store prediction of test sample
predict_svm = zeros(sample_size, 1);

for i = 1: sample_size
    for j = 1: num_labels
        % label = num2str(j);
        label_predicate = predict(models_svm{j, 1}, X_test(i,:));
        if(label_predicate == 1)
            Scores(i, j) = Scores(i, j) + 1;
        end
        if(label_predicate == 0)
            Scores(i, j) = Scores(i, j) - 1;
        end
    end
end
% [~,maxScore] = max(Scores,[],2);

for i = 1 :1: sample_size
    max = Scores(i, 1);
    predict_svm(i, 1) = 1;
    for j = 1 :1: num_labels
        if(max < Scores(i, j))
            max = Scores(i, j);
            predict_svm(i, 1) = j;
        end
    end
end

res_svm = predict_svm - y_test;

% number of matching
match_svm = 0;

for i = 1:1:sample_size
    % count number of matching
    if res_svm(i,1) == 0
        match_svm = match_svm + 1;
    end
end

% percentage of matching
per_svm = match_svm / sample_size * 100;
fprintf('SVM Precsion is %.2f%%. \n', per_svm);

% necessary parameter
NUM_SAMPLE = size(X_train, 1);
NUM_TEST_SAMPLE = size(X_test, 1);
NUM_NEURON = 25;

size_y_train = size(y_train);
size_y_test = size(y_test);
y_predicts = zeros(NUM_SAMPLE,NUM_NEURON);

% precision of training set
for i = 1:size_y_train
    for j = 1:NUM_NEURON
        if y_train(i, 1) == j
            y_predicts(i, j) = 1;
        end
    end
end

% training model using neural network
net = feedforwardnet(NUM_NEURON);
%FOR testing
%net.trainParam.epochs = 1;

net = configure(net, X_train', y_predicts'); view(net)
net = train(net,X_train', y_predicts'); view(net)

res_ANN = net(X_test');
res_ANN = res_ANN';
all_predicts = zeros(NUM_TEST_SAMPLE,1);

% precision of test sample
for i = 1:size_y_test
    max = realmin('double');
    for j = 1:NUM_NEURON
        if max < res_ANN(i, j)
            max = res_ANN(i, j);
            predict = j;
        end
    end
    all_predicts(i,1) = predict;
end

% difference between precision and actual result
diff_predicts = all_predicts - y_test;

% number of matching
num_match = 0;

% count number of matching
for i = 1:1:size_y_test
    if diff_predicts(i,1)==0
        num_match = num_match+1;
    end
end

% reporting
precision = (num_match/NUM_TEST_SAMPLE) * 100;
fprintf('ANN Precision is %.2f%%. \n',precision);

final_predict=zeros(sample_size, 1);
for i=1:sample_size
    predict_table(i, predict_knn(i, 1))=predict_table(i, predict_knn(i, 1))+1;
    predict_table(i, predict_svm(i, 1))=predict_table(i, predict_svm(i, 1))+1;
    predict_table(i, all_predicts(i, 1))=predict_table(i, all_predicts(i, 1))+1;
    max=0;
    max_label=0;
    for j=1:num_classes
        if predict_table(i, j) > max
            max_label=j;
            max=predict_table(i, j);
        end;
    end;
    if max==1
        final_predict(i)=predict_knn(i, 1);
    else
        final_predict(i)=max_label;
    end;
end;

acc = sum(final_predict == y_test) / size(y_test, 1) * 100;
% fprintf('The accuracy using k-nearest neighbor with k = 5 is %.2f%%. \n', per_knn * 100);
fprintf('Ensemble Precsion is %.2f%%. \n', acc);




