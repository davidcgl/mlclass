function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

% Number of training examples and labels
m = size(X, 1);
num_labels = size(all_theta, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% m x num_label matrix, where:
% probs(i,j) = probability that example i belongs to label j  
probs = X * all_theta';

% Get the highest probabilities from each row and save its index (label) into p 
[highest_probs p] = max(probs, [], 2);

end
