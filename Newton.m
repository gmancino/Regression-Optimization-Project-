%-----------------------------------
% Newton's Method for Logistic Regression
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
%
% Outputs:
% w - normal vector for hyperplane
% b - scalar for hyperplane
% hist_obj - history of objective value
% iter - number of iterations
%-----------------------------------
function [w, b, iter, hist_obj] =...
    Newton(X, y, w, b, lambda1, lambda2, maxit, tol)
    [m,n] = size(X);
    iter = 1;
    grad = UpdatedGradLR(X, y, w, b, lambda1, lambda2);
    H = UpdatedHessLR(X, y, w, b, lambda1, lambda2);
    obj = UpdatedObjLR(X, y, w, b, lambda1, lambda2);
    hist_obj = obj;
    % Pure Newton's
    alpha = 1;
    
    while iter < maxit && norm(grad(1:n,1))+norm(grad(n+1,1))...
            >= tol*max(1, norm(w)+norm(b))
        
        % Find descent direction
        d = linsolve(H, grad);
        
        % Update w and b
        w = w - alpha*d(1:n,1);
        b = b - alpha*d(n+1,1);
        
        % Update hessian and gradient
        H = UpdatedHessLR(X, y, w, b, lambda1, lambda2);
        grad = UpdatedGradLR(X, y, w, b, lambda1, lambda2);
        
        % Update objective
        obj = UpdatedObjLR(X, y, w, b, lambda1, lambda2);
        hist_obj = [hist_obj; obj];
        iter = iter + 1;
    end
end