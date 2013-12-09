function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples and features
m = size(X, 1);
n = size(X, 2);

% Training and cross validation errors for each training set size 1:m
error_train = zeros(m, 1);
error_val = zeros(m, 1);

% No regularization term for calculating training and cross validation errors
lambda_error = 0;

% Train the linear regression classifier with different size of training set
% and calculate the training error (J_train) and cross validation error (J_cv)
% of the trained model
for i = 1:m
    % Get a subset of training examples
    X_train = X(1:i, :);
    y_train = y(1:i);

    % Train the classifier and calculate errors
    theta = trainLinearReg(X_train, y_train, lambda);
    error_train(i) = linearRegCostFunction(X_train, y_train, theta, lambda_error);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, lambda_error);
end

end