function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Number of training examples
m = length(y);                      

% Distance between actual value and predicted value of each training example
errors = (X * theta - y);

% Calculate cost function J(theta) using sum of squared errors
J = 1/(2*m) * (errors' * errors);

end
