function [RI, ARI] = randomIndex(trueLabels, predictedLabels)
% Performance assesment of clustering problems. The function gets the true
% labels (trueLabels) and the predicted labels (predictedLabels) and
% outputs their similarity. RI, random index, is the similarity measure (check the
% internet for a better description) and ARI, adjusted RI, is some
% normalization of it making sure that random similarity between trueLabels
% and predictedLabels will give 0.0 (for both RI and ARI perfect
% agreement=1.0)
% ** By the way, the funcion is symetric. Hence swaping the input vectors will not change the outcome 
    % Ensure the input vectors are of the same length
    if length(trueLabels) ~= length(predictedLabels)
        error('The length of label vectors must be the same');
    end

    % Create a contingency table
    C = contingency_table(trueLabels, predictedLabels);

    % Sum over combinations of pairs in each cell
    sum_comb = @(x) sum(x .* (x - 1) / 2, 'all');
    sumC2 = sum_comb(C);

    % Sum over rows and columns, and then get the combinations
    sumR2 = sum_comb(sum(C, 2));
    sumS2 = sum_comb(sum(C, 1));

    % Total pairs
    n = length(trueLabels);
    totalPairs = n * (n - 1) / 2;

    % Calculate Rand Index
    RI = (sumC2 + (totalPairs - sumR2 - sumS2+sumC2)) / totalPairs;

    % Calculate Adjusted Rand Index
    expectedRI = (sumR2 * sumS2) / totalPairs;
    maxRI = 0.5 * (sumR2 + sumS2);
    ARI = (sumC2 - expectedRI) / (maxRI - expectedRI);
    ARI=full(ARI);
%     expectedRI = (sumR2 * sumS2) / totalPairs;
%     maxRI = (sumR2 + sumS2) / 2;
%     ARI = (RI - expectedRI) / (maxRI - expectedRI);

end

function C = contingency_table(a, b)
    % Create a sparse matrix with counts
    C = sparse(a, b, 1);
end