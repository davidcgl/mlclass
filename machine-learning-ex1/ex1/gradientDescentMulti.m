function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%

% Number of training examples.
m = length(y);

% For plotting J vs time graph.
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % See derivation in handout.
    delta = X' * (X * theta - y);

    % Perform gradient descent - update value of theta.
    theta = theta - (alpha / m) * delta;

    % Save the cost J in every iteration.
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
