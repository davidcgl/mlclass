function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i

% Number of training examples and features
m = size(X, 1);
n = size(X, 2);

X = [ones(m, 1) X];
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Each row all_theta(i) correspond to theta parameters that minimizes the cost
% J(all_theta(i)) = local optima for label i
all_theta = zeros(num_labels, n+1);

for c = 1:num_labels
    % m-dimensional vector of {0,1}, where:
    %   0 = y(i) belongs to label c
    %   1 = y(i) does not belong to label c
    pos_examples = y == c;

    initial_theta = zeros(n+1, 1);

    % Calculate the optimal theta for label i
    all_theta(c, :) = ...
        fmincg(@(t)(lrCostFunction(t, X, pos_examples, lambda)), initial_theta, options);
end

end
