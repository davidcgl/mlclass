function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Number of training examples
m = size(X, 1);

% Reshape nn_params back into the parameters Theta1 and Theta2, 
% the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% -----------------
% Matrix dimensions 
% -----------------
% X         => input_layer_size * m
% y         => num_labels * m
% Theta1    => hidden_layer_size * input_layer_size+1
% Theta2    => num_labels * hidden_layer_size+1

for i = 1:m
    % =========================================================================
    % Feed forward
    % =========================================================================

    % Input layer
    a1 = [1; X(i, :)'];

    % Feed forward from input layer to layer 2
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];

    % Feed forward from layer 2 to output layer
    z3 = Theta2 * a2; 
    a3 = sigmoid(z3);

    % Map output y(i) into binary vector of {0,1}
    % For example:
    %   Let num_labels = 3 and y(i) = 2
    %   Then y_vec = [0; 1; 0]
    y_vec = zeros(num_labels, 1);
    y_vec(y(i)) = 1;

    % Calculate prediction costs for each output unit a3(i), where:
    %   Prediction cost = cost of having sigmoid(a3(i)) given value of y_vec(i)
    prediction_costs = (-y_vec .* log(a3)) - ((1-y_vec) .* log(1-a3));
    total_prediction_costs = sum(sum(prediction_costs));

    % Accumulate cost J
    J += (1/m) .* total_prediction_costs;

    % =========================================================================
    % Backpropagation
    % =========================================================================

    % Calculate error terms for each unit on layer 2 and layer 3 (output)
    delta3 = a3 - y_vec;
    delta2 = (Theta2(:, 2:end)' * delta3) .* sigmoidGradient(z2);

    % Accumulate error terms
    Theta1_grad += (delta2 * a1');
    Theta2_grad += (delta3 * a2');
end

% Calculate regularized cost function J
Theta1_sum_square = sum(sum(Theta1(:, 2:end) .** 2));
Theta2_sum_square = sum(sum(Theta2(:, 2:end) .** 2));
regterm = (lambda / (2*m)) .* (Theta1_sum_square + Theta2_sum_square);
J += regterm;

% Calculate gradient of J (with regularization)
Theta1_grad = (1/m) .* Theta1_grad;
Theta1_grad(:, 2:end) += ((lambda/m) .* Theta1(:, 2:end));

Theta2_grad = (1/m) .* Theta2_grad;
Theta2_grad(:, 2:end) += ((lambda/m) .* Theta2(:, 2:end));

% Return gradients as a column vector
grad = [Theta1_grad(:); Theta2_grad(:)]; 

end
