function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Number of training examples
m = length(y);

% Cost J at each iteration
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta = theta - (alpha/m) * (X' * (X*theta - y));
    J_history(iter) = computeCostMulti(X, y, theta);
end

end