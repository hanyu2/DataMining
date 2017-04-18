function [mag] = get_magnitude(v1)

% GET_MAGNITUDE: This function calculates the magnitude of a given vector
% v1.

% Input
%=======
% v1  -> The input vector

% Output
%========
% mag -> The magnitude of the vector

% Written, Shayok, March 2009
%==========================================================================

 n = length(v1);  %% dimension of the vector
 
 sum = 0;  %% initialise summation
 
 for i = 1:1:n
     
  sum = sum + v1(i)*v1(i);  %% squared sum
  
 end  %% end for
 
 mag = sqrt(sum);  %% magnitude of the vector
 
end  %% end function