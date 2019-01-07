%-----------------------------------
% Classification Function for Logistic Regression
%
% Inputs:
% X(i,:) - ith data point
% w - normal vector for hyperplane
% b - scalar for hyperplane
%
% Outputs:
% y - vector with -1 and 1 as components
%-----------------------------------
function y = ClassLR(X, w, b)
    [m,n] = size(X);
    y = zeros(m,1);
    for i=1:m
       if X(i,:)*w+b >= 0
           y(i,1) = 1;
       else
           y(i,1) = -1;
       end
    end
end