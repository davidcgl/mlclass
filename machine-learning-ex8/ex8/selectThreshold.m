function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = (min(pval) + stepsize):stepsize:max(pval)
    % Predict anomaly if pval < epsilon.
    predictions = pval < epsilon;

    tpos = sum((predictions == 1) & (yval == 1));
    fpos = sum((predictions == 1) & (yval == 0));
    fneg = sum((predictions == 0) & (yval == 1));

    precision = tpos / (tpos + fpos);
    recall = tpos / (tpos + fneg);
    F1 = (2 * precision * recall) / (precision + recall);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
