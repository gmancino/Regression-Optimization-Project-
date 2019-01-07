%-----------------------------------
% Hessian of Objective Function for LR
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
% H - hessian matrix of f
%-----------------------------------
function H = UpdatedHessLR(X, y, w, b, lambda1, lambda2)
    [m,n] = size(X);
    H = zeros(n+1,n+1);
    evec = exp(y.*(X*w+b))./(1+exp(y.*(X*w+b))).^2;
    diage = diag(evec);
    H(1:n,1:n) = (1/m)*(X'*diage)*X + eye(n)*lambda1;
    H(1:n,n+1) = (1/m)*sum(X.*evec)';
    H(n+1,1:n) = H(1:n,n+1)';
    H(n+1,n+1) = (1/m)*sum(evec) + lambda2;
end