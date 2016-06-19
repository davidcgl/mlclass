function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

% Add a bias unit to input layer.
% Forward propagate from input layer to layer 2.
layer1 = [ones(1, m) X];
layer2 = sigmoid(layer1 * Theta1');

% Add a bias unit to layer 2.
% Forward propagate from layer 2 to layer 3.
layer2 = [ones(1, size(layer2, 2)) layer2];
layer3 = sigmoid(layer2 * Theta2');

% layer3 is m x 10 matrix.
% For each row, maximum value's index (1-10) is the predicted class.
[max_val, max_val_indices] = max(layer3, [], 2);
p = max_val_indices;

end
