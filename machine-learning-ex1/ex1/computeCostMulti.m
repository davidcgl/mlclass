function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Number of training examples.
m = length(y);

% Sum of squared errors.
errors = (X * theta - y);
sumSqErrors = errors' * errors;

% Total cost.
J = (1 / (2*m)) .* sumSqErrors; 

end
