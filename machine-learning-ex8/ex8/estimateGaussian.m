function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% m = number of examples
% n = number of features
[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

for i = 1:n
  mu(i) = (1 / m) * sum(X(:, i)); 
  sigma2(i) = (1 / m) * sum((X(:, i) - mu(i)) .^ 2);
end

end
