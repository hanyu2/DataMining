function [H,y] = get_second_derivative(training_matrix,weight_current,lambda)


% GET_SECOND_DERIVATIVE: This function calculates the second derivative of
% the error as a function of w

% Input
%========
% training_matrix  -> The matrix of training instances
% weight_current   -> The current weight vector
% lambda           -> The regularizing parameter

% Output
%=========
% H      -> The Hessian matrix
% y      -> The vector of predictions

% Written, Shayok, March, 2009
%==========================================================================


 [row column] = size(training_matrix);  %% dimensions of the training matrix
 
 H = zeros(column,column);  %% initialise Hessian matrix
 
 y = [];  %% initialise a zero vector
 
 for i = 1:1:row  %% for each training point
     
     phi_n = [];
     phi_n = training_matrix(i,:)';  %% get phi_n
     
     a_n = weight_current'*phi_n;  %% get a_n 
      
     y_n = 1 / (1 + exp(-a_n));  %% get y_n
     
     y(i) = y_n;  %% populate the y vector of predictions
     
     H = H + y_n * (1 - y_n) * phi_n * phi_n' + lambda * eye(column);  %% keep calculating the Hessian
     
 end  %% end for

end  %% end function
