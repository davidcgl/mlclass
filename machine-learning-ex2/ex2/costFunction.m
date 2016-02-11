function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y);

% P(y = 1 | x; theta)
pred = sigmoid(X * theta);

% Calculate total cost of mispredicting positive / negative examples using
% vectorized formula.
posCost = log(pred)' * y;
negCost = log(1 - pred)'  * (1 - y);
J = (-1 / m) .* (posCost + negCost);

% Calculate gradient at current thetat using vectorized formula.
grad = (1 / m) .* (X' * (pred - y));

end
