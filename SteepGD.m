%-----------------------------------
% Steepest Gradient Descent for Logistic Regression
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
    SteepGD(X, y, w, b, lambda1, lambda2, maxit, tol)
    [m,n] = size(X);
    iter = 1;
    grad = UpdatedGradLR(X, y, w, b, lambda1, lambda2);
    obj = UpdatedObjLR(X, y, w, b, lambda1, lambda2);
    hist_obj = obj;
    while iter < maxit && norm(grad(1:n,1))+norm(grad(n+1,1))...
            >= tol*max(1, norm(w)+norm(b))
 
        % Loop for alpha using backtracking
        alpha = 1;
        while UpdatedObjLR(X, y, w - alpha*grad(1:n,1),...
                b - alpha*grad(n+1,1), lambda1, lambda2) - ...
                UpdatedObjLR(X, y, w, b, lambda1, lambda2) >= 0.5*alpha*...
                grad'*(-grad)
            alpha = 0.75*alpha;
        end
    
        % Update w and b
        w = w - alpha*grad(1:n,1);
        b = b - alpha*grad(n+1,1);
        
        % Update gradient
        grad = UpdatedGradLR(X, y, w, b, lambda1, lambda2);
    
        % Update objective
        obj = UpdatedObjLR(X, y, w, b, lambda1, lambda2);
        hist_obj = [hist_obj; obj];
        iter = iter + 1;
    end
end