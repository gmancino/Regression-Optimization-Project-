%-----------------------------------
% Objective Function for LR
% (vectorized)
%
% Inputs:
% X(i,:) - ith data point
% y - vector of classification results
% w - normal vector to hyperplane
% b - scalar in hyperplane equation
% lambda1 - tuning parameter
% lambda2 - tuning parameter
%
% Outputs:
% f - function value
%-----------------------------------
function f = UpdatedObjLR(X, y, w, b, lambda1, lambda2)
    [m,n] = size(X);
    f = 1/m*(sum(log(1+exp(-y.*(X*w+b)))))+0.5*lambda1*(w'*w)+0.5*lambda2*b^2;
end