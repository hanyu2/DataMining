load('X_test')
load('X_train')
load('y_test')
load('y_train')

kernel = 2;
n=size(X_train, 1);
label_num=size(y_test, 2);
models_svm = cell(label_num, 1);
for i=1:label_num
    SVMModel=fitcsvm(X_train, y_train(:,i), 'KernelScale', 'auto','KernelFunction', 'polynomial', 'PolynomialOrder',kernel, 'NumPrint', 10000);
    models_svm{i, 1}=SVMModel;
end;

my_predict=zeros(size(y_test, 1), size(y_test, 2));
for i=1:size(X_test, 1)
    for j = 1: label_num
        label_predicate = predict(models_svm{j, 1}, X_test(i,:));
        if label_predicate > 0
            my_predict(i, j)=1;
        end;
        if label_predicate < 0
            my_predict(i, j)=0;
        end;
    end;
end;

acc = 0;
both= 0;
total = 0;
for i=1:size(X_test, 1)
    for j = 1: label_num
        if my_predict(i, j)==1 || y_test(i, j)==1
            total=total+1;
        end;
        if my_predict(i, j)==1 && y_test(i, j)==1
            both=both+1;
        end;
    end;
end;
fprintf('Polynomial accuracy is %.2f%%. \n', 100*both/total);
    
for i=1:label_num
    SVMModel=fitcsvm(X_train, y_train(:,i), 'KernelScale', 'auto','KernelFunction', 'gaussian', 'NumPrint', 10000);
    models_svm{i, 1}=SVMModel;
end;
my_predict=zeros(size(y_test, 1), size(y_test, 2));
for i=1:size(X_test, 1)
    for j = 1: label_num
        label_predicate = predict(models_svm{j, 1}, X_test(i,:));
        if label_predicate > 0
            my_predict(i, j)=1;
        end;
        if label_predicate < 0
            my_predict(i, j)=0;
        end;
    end;
end;

acc = 0;
both= 0;
total = 0;
for i=1:size(X_test, 1)
    for j = 1: label_num
        if my_predict(i, j)==1 || y_test(i, j)==1
            total=total+1;
        end;
        if my_predict(i, j)==1 && y_test(i, j)==1
            both=both+1;
        end;
    end;
end;
fprintf('Gaussian accuracy is %.2f%%. \n', 100*both/total);
    