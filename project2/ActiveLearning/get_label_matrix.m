function [label_matrix] = get_label_matrix(training_label,num_classes)

% GET_LABEL_MATRIX: This function creates a label matrix from the training
% label vector. For a particular class, only the corresponding entry inb
% the matrix is 1, all other entries are 0s. 

%==========================================================================

len = length(training_label);
label_matrix = zeros(len,num_classes);


for i = 1:1:len
    
    label = training_label(i);
    label_matrix(i,label) = 1;
    
end  %% end for


end  %% end function