function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Number of training examples.
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Compute slope of J(theta) at each dimension n via partial derivatives.
    % This is more efficient in vectorized form.
    delta = X' * (X * theta - y);

    % Perform gradient descent on theta.
    % If delta_n > 0, then theta_n decreases.
    % If delta_n < 0, then theta_n increases.
    theta = theta - (alpha / m) * delta;

    % Save J(theta) during each iteration. This is to check that J(theta)
    % decreases over time, and how fast J(theta) converges to minima.
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
