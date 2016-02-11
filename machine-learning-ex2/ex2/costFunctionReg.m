function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);

% Calculate cost and gradient without regularization.
[J, grad] = costFunction(theta, X, y);

J_reg =  (lambda / (2 * m)) .* (theta(2:end)' * theta(2:end));
J += J_reg;

grad_reg = (lambda /  m) .* [0; theta(2:end)];
grad += grad_reg;

end
