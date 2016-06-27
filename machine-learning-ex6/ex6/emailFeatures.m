function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% x is a nx1 feature vector, such that x_i = 1 if word_indices contains index i
% which corresponds to vocab i in the dictionary.
x = zeros(n, 1);

for i = 1:length(word_indices)
  x(word_indices(i)) = 1;
end

end
