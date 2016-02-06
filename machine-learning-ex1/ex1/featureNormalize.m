function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Number of features.
n = size(X, 2);

X_norm = X;
mu = zeros(1, n);
sigma = zeros(1, n);

for j=1:n
  % Find mean and std dev of feature_j.
  X_j = X(:,j);
  mu(j) = mean(X_j);
  sigma(j) = std(X_j);

  % Normalize feature_j values.
  X_norm(:,j) = (X_j - mu(j)) / sigma(j);
end

end
