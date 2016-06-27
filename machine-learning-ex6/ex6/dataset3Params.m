function [C, sigma] = dataset3Params(X, y, Xval, yval)
%dataset3Params returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel

% Best parameters found from the following code.
C = 1;
sigma = 0.1;

% C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% prediction_error = flintmax();
% 
% for i = 1:length(C_vec)
%   for j = 1:length(sigma_vec)
%     fprintf('Training SVM with C = %f, sigma = %f\n', C_vec(i), sigma_vec(j));
%     model = svmTrain(X, y, C_vec(i), ...
%         @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j))); 
% 
%     predictions = svmPredict(model, Xval);
%     prediction_error_cand = mean(double(predictions != yval));
% 
%     if prediction_error_cand < prediction_error
%       C = C_vec(i);
%       sigma = sigma_vec(j);
%       prediction_error = prediction_error_cand;
%     end
%   end
% end
% 
% fprintf('Best parameters: C = %f, sigma = %f\n', C, sigma);

end
