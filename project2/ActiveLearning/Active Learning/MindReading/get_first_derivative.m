function [delta_error] = get_first_derivative(training_matrix,label_vector,weight_current,lambda)

% GET_FIRST_DERIVATIVE: This function calculates the first derivative of
% the error as a function of w

% Input
%========
% training_matrix  -> The matrix of training instances
% label_vector     -> The vector of class labels
% weight_current   -> The current weight vector
% lambda           -> The regularizing parameter

% Output
%=========
% delta_error      -> The vector of first derivative of error function

% Written, Shayok, March, 2009
%==========================================================================

 [row column] = size(training_matrix);  %% dimensions of the training matrix
 
 delta_error = zeros(column,1);  %% initialise vector
 
  for i = 1:1:row  %% for each training point
      
      phi_n = [];
      phi_n = training_matrix(i,:)';  %% get phi_n
      
      a_n = weight_current'*phi_n;  %% get a_n 
      
      y_n = 1 / (1 + exp(-a_n));  %% get y_n
      
      t_n = label_vector(i);  %% current class label
      
      delta_error = delta_error + (y_n - t_n) * phi_n + lambda*weight_current;  %% keep calculating the derivative
      
  end  %% end for

end  %% end function