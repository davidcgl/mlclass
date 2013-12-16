function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Candidates for parameters C and sigma
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100];

% Minimum prediction error for choosing C and sigma
min_pred_error = realmax;

for i = 1:length(C_vec)
    C_train = C_vec(i);

    for j = 1:length(sigma_vec)
        sigma_train = sigma_vec(j);

        % Train the SVM model
        model = svmTrain(X, y, C_train, @(x1, x2) gaussianKernel(x1, x2, sigma_train));

        % Make predictions using the trained model
        pred = svmPredict(model, Xval);

        % Calculate the prediction accuracy
        pred_error = mean(double(pred ~= yval));

        if pred_error < min_pred_error
            C = C_train;
            sigma = sigma_train;
            min_pred_error = pred_error;
        end
    end
end

end
