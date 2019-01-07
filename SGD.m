%-----------------------------------
% Stochastic Gradient Descent for Logistic Regression
%
% Inputs:
% X(i,:) - ith data point as a row vector
% y - {-1, +1} classifier
% w - initial guess for w
% b - initial guess for b
% lambda1 - tuning parameter
% lambda2 - tuning parameter
% maxit - max number of iteration
% tol - tolerance
% batch - batchsize
%
% Outputs:
% w - normal vector for hyperplane
% b - scalar for hyperplane
% hist_obj - history of objective value
% iter - number of iterations
%-----------------------------------
function [w, b, iter, hist_obj] =...
    SGD(X, y, w, b, lambda1, lambda2, maxit, batch)
    [m,n] = size(X);
    hist_obj = 0;
    iter = 1;
    while iter < maxit
    for i=1:(m/batch)
        % Pick random entries
        r = randi([1,m], batch, 1);
        Xup = X(r,:);
        yup = y(r,1);
        sgrad = UpdatedGradLR(Xup, yup, w, b, lambda1, lambda2);
        
        % Set alpha
        alpha = (5/6)/(iter);
    
        % Update w and b
        w = w - alpha*sgrad(1:n,1);
        b = b - alpha*sgrad(n+1,1);
        
    end
    % Update objective only if obj != Inf
    obj = UpdatedObjLR(X, y, w, b, lambda1, lambda2);
    if obj~=Inf
    hist_obj = [hist_obj; (sum(hist_obj) + obj)/iter];
    else
    hist_obj = [hist_obj; hist_obj(length(hist_obj))];
    end
    iter = iter + 1;
    end
end