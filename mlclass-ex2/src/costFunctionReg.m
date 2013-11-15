function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Number of training examples
m = length(y);

% Number of features
n = length(theta);

% Calculate the sigmoid value (score) of each training example
% score(x) = P(Y=1 | x;theta)
scores = sigmoid(X * theta);

% Regularization term
regterm = (lambda/(2*m)) .* (theta(2:n)' * theta(2:n));

% Total cost J
J = ((-1/m) * sum(y .* log(scores) + (1-y) .* log(1-scores))) + regterm;

% Regularization term for gradient
regterm_grad = (lambda/m) .* [0; theta(2:n)];

% Gradient of J
grad = ((1/m) .* (X' * (scores-y))) + regterm_grad;

end
