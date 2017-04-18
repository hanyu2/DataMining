function [final_weight] = train_classifier(training_matrix,label_vector,epsilon,max_iterations,lambda)

% TRAIN_CLASSIFIER: This function trains the logistic regression classifier

% Input
%========
% training_matrix  -> The matrix of training data points
% label_vector     -> The class labels of the training points
% epsilon          -> Threshold parameter
% max_iterations   -> The maximum number of iterations
% lambda           -> The regularization parameter

% Output
%========
% final_weight     -> The final value of the weight vector

% Written, Shayok, March, 2009
%==========================================================================

 [row column] = size(training_matrix);  %% dimensions of the training data
  
  w_old = zeros(column,1);  %% initialise a zero vector as weights
  y_previous = zeros(row,1);  %% initialise a zero vector for predictions
  num_iterations = 1;  %% initialize number of iterations
  
  while(1)  %% infinite while loop
      
      [delta_error] = get_first_derivative(training_matrix,label_vector,w_old,lambda);  %% get delta E(w)
      
      [H,y_current] = get_second_derivative(training_matrix,w_old,lambda);  %% get the Hessian matrix
      
      w_new = [];  
      w_new = w_old - inv(H) * delta_error;   %% update the weight

      num_iterations = num_iterations + 1;  %% increment counter  
      
      if (num_iterations > max_iterations)  %% breaking condition
          break;
      end
      
      [mag_previous] = get_magnitude(y_previous); 
      [mag_current] = get_magnitude(y_current);  %% get magnitudes of the prediction vectors
      
      if abs(mag_previous - mag_current) < epsilon  %% another breaking condition
          break;
      end
      
      w_old = [];
      w_old = w_new;  %% change the old vector to the new one
      
      y_previous = [];
      y_previous = y_current;  %% change the old vector to the new one
      
  end  %% end while
  
  final_weight = w_new;  %% return the trained weight vector
  
end  %% end function