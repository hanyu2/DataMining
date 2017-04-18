function [probabilityVector] = test_LR_Classifier(testSample,trained_weights,num_classes)

% This function computes the probability of a test sample corresponding to
% all the classes using the trained logistic regression classifier

% Input
%========
% testSample        -> The test sample (row vector)
% trained_weights   -> The weights of the trained LR model (cell array)
% num_classes       -> The possible number of classes

% Output
%========
% probabilityVector -> A vector containing the class probabilities for the
%                     test sample
%==========================================================================
    
prob = [];  %% clear

for j = 1:1:num_classes

    weight = trained_weights{j};  %% retrive the weight for the class

    y = weight'*testSample';  %% multiply by the weight vector

    prob(j) = exp(y); %% compute the probability

end  %% end inner for

probabilityVector = [];
probabilityVector = prob / sum(prob); %% normalize the probability values

end  %% end function 