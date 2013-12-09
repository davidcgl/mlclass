function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Training and cross validation errors for each value of lambda
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% No regularization term for calculating training and cross validation errors
lambda_error = 0;

% Train linear regression classifier using different lambda values
% and calculate the training error (J_train) and cross validation error (J_cv)
% of the trained model
for i = 1:length(lambda_vec)
    theta = trainLinearReg(X, y, lambda_vec(i));
    error_train(i) = linearRegCostFunction(X, y, theta, lambda_error);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, lambda_error);
end

end
