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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1_row = hidden_layer_size;
Theta1_col = input_layer_size + 1;
Theta1_size = Theta1_row * Theta1_col;
Theta1 = reshape(nn_params(1:Theta1_size), Theta1_row, Theta1_col);

Theta2_row = num_labels;
Theta2_col = hidden_layer_size + 1;
Theta2_size = Theta2_row * Theta2_col;
Theta2 = reshape(nn_params((Theta1_size + 1):end), Theta2_row, Theta2_col);

% Number of input.
m = size(X, 1);

% Cost function and gradient.
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
  % Represent y in vector format: [1 0 0] == 1, [0 1 0] == 2, etc
  y_i = zeros(1, num_labels);
  y_i(y(i)) = 1;

  % ============================================================================
  % Forward propagation
  % ============================================================================

  % Layer 1
  x = X(i, :);
  a1 = [1 x];

  % Layer 1 => Layer 2
  z2 = a1 * Theta1';
  a2 = [1 sigmoid(z2)];

  % Layer 2 => Layer 3
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  % Accumulate total cost.
  J = J + sum(y_i .* log(a3) + (1 - y_i) .* log(1 - a3));

  % ============================================================================
  % Back propagation
  % ============================================================================

  % Layer 3
  d3 = a3 - y_i;

  % Layer 3 => Layer 2
  d2 = (d3 * Theta2);
  d2 = d2(2:end) .* sigmoidGradient(z2);

  % Accumulate gradients to calculate partial derivatives of each weight.
  Theta1_grad = Theta1_grad + (d2' * a1); 
  Theta2_grad = Theta2_grad + (d3' * a2);  
end;

% ==============================================================================
% Regularization
% ==============================================================================

J = (-J / m) + (lambda / (2 * m)) ...
    * ((sum(sum(Theta1(:, 2:end).^ 2))) + (sum(sum(Theta2(:, 2:end) .^ 2))));

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda * Theta2(:, 2:end);

% Unroll gradients
grad = (1 / m) * [Theta1_grad(:) ; Theta2_grad(:)];

end
