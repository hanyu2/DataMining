function [trained_weights] = train_LR_Classifier(training_matrix,training_label,num_classes)

% This function trains a logistic regression classification model

% Input
%========
% training_matrix    -> The matrix of training data points (each row is a
%                       sample and each column is a feature)
% training_label     -> The class labels of the training points
% num_classes        -> The possible number of classes

% Output
%========
% trained_weights    -> The weights of the trained logistic regression
%                       classifier (cell array)
%==========================================================================

epsilon = 0.0001;
max_iterations = 200;  %% parameters of the LR model
lambda = 0.01; 

trained_weights = []; %% initialize to null

[label_matrix] = get_label_matrix(training_label,num_classes); %% compute the binary label matrix

for i = 1:1:num_classes
    
    label_vector = label_matrix(:,i); %% train the LR model for a given class

    [final_weight] = train_classifier(training_matrix,label_vector,epsilon,max_iterations,lambda);  %% final trained weights
      
    trained_weights{i} = final_weight;  %% store the trained weights
    
end  %% end for

end  %% end function 