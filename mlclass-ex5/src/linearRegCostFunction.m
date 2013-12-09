function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Number of training examples and features
m = length(y);
n = length(theta);

% Distance between actual value and predicted value of each training example
errors = (X * theta - y);

% Regularization term
regterm = (lambda/(2*m)) .* (theta(2:n)' * theta(2:n));

% Total cost J
J = (1/(2*m)) * (errors' * errors) + regterm;

% Regularization term for gradient
regterm_grad = (lambda/m) .* [0; theta(2:n)];

% Gradient of J
grad = (1/m) .* (X' * errors) + regterm_grad;

end
