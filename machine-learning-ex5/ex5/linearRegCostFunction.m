function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Number of training examples.
m = length(y);

% Sum squared errors.
errors = X * theta - y;
sumSqErrors = errors' * errors;

J_reg = lambda * (theta(2:end)' * theta(2:end));
J = (1 / (2 * m)) * (sumSqErrors + J_reg);

grad_reg = lambda * [0; theta(2:end)];
grad = (1 / m) * (X' * errors + grad_reg);

end
