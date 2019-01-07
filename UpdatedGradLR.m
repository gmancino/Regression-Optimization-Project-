%-----------------------------------
% Gradient of Objective Function for LR
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
% gradf - gradient of function evaluated at w and b
%-----------------------------------
function gradf = UpdatedGradLR(X, y, w, b, lambda1, lambda2)
    [m,n] = size(X);
    gradf = zeros(n+1,1);
    gradf(1:n,1) = 1/m*sum((-y.*X./(exp(y.*(X*w+b))+1)))' + lambda1*w;
    gradf(n+1,1) = 1/m*sum((-y./(exp(y.*(X*w+b))+1))) + lambda2*b;
end