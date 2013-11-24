function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Number of training examples and features
m = length(y);
n = length(theta);

% Calculate the sigmoid value (score) of each training example
% score(x) = P(Y=1 | x;theta)
scores = sigmoid(X * theta);

% Regularization term for logistic regression cost J and gradient
regterm = (lambda/(2*m)) .* (theta(2:n)' * theta(2:n));
regterm_grad = (lambda/m) .* [0; theta(2:n)];

% Calculate J and its gradient
J = ((-1/m) * sum(y .* log(scores) + (1-y) .* log(1-scores))) + regterm;
grad = ((1/m) .* (X' * (scores-y))) + regterm_grad;

end
