function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Number of training examples
m = length(y);

% Calculate the sigmoid value (score) of each training example
% score(x) = P(Y=1 | x;theta)
scores = sigmoid(X * theta);

% Total cost J
J = (-1/m) * sum(y .* log(scores) + (1-y) .* log(1-scores));

% Gradient of J
grad = (1/m) .* (X' * (scores - y));

end
