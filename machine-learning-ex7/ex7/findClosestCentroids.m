function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Number of centroids.
K = size(centroids, 1);

% Number of examples.
m = size(X, 1);

% idx(i) == k means training example i is assigned to centroid k.
idx = zeros(m, 1);

% Assign each training example to the closest centroid [1..K].
for i = 1:m
  x_i = X(i, :);
  minDistance = intmax();
  for j = 1:K
    c_i = centroids(j, :);
    distance = norm(x_i - c_i) ^ 2;
    if distance < minDistance
      idx(i) = j;
      minDistance = distance;
    end
  end 
end

end

