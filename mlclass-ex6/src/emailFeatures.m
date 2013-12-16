function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% Vocabulary list for checking word existence
vocab_list = getVocabList();

% If x(i) exists in word_indices, then x(i) is set to 1
x = zeros(n, 1);
x(word_indices) = 1;

end
