function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Number of training examples, labels
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add bias unit to input layer
X = [ones(m, 1) X];

% Feed forward layer 1 => layer 2
z2 = Theta1 *  X';
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];

% Feed forward layer 2 => layer 3
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% a3 is now a num_labels x m matrix, where:
% a3(i,j) = probability that example j belongs to label i

% Get the highest probabilities from each row and save its index (label) into p 
[highest_probs p] = max(a3', [], 2);

end
