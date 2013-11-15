function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Sample mean and std deviations of each feature in training examples
mu = mean(X);
sigma = std(X);

% Normalized values of each feature in training examples
X_norm = (X - mu) ./ sigma;

end