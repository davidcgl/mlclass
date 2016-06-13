function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
J = 0;
grad = zeros(size(theta));

% P(y = 1 | x; theta)
pred = sigmoid(X * theta);

% Calculate total cost of mispredicting positive / negative examples
% with regularization term for all theta except theta_0.
posCost = log(pred)' * y;
negCost = log(1 - pred)'  * (1 - y);
J = (-1 / m) .* (posCost + negCost);
J = J + (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));

% Calculate gradient with regularization term for all theta except theta_0.
grad = (1 / m) .* (X' * (pred - y));
grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);

end
