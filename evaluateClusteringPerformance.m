function [external, internal, stability] = evaluateClusteringPerformance(X, true_labels, idx_mat)
%==========================================================
% evaluateClusteringPerformance
% ----------------------------------------------------------
% Evaluates external and internal clustering metrics, and computes
% cluster stability via Hungarian matching alignment.
%
% INPUTS:
%   X           - Feature matrix (N × d)
%   true_labels - Ground-truth class labels (N × 1)
%   idx_mat     - Matrix of clustering results (N × R),
%                 where each column is one clustering realization
%
% OUTPUTS:
%   external    - Struct containing RI and ARI values and their means
%   internal    - Struct containing Compactness, Dunn, and Silhouette metrics
%   stability   - Scalar stability score computed using Hungarian matching
%==========================================================

    %-----------------------------
    % External clustering metrics
    %-----------------------------
    RI_mat=zeros(1,10);
    ARI_mat=zeros(1,10);
    for i=1:10
        [RI_mat(i),ARI_mat(i)] = randomIndex(true_labels, idx_mat(:,i));
    end

    % Create structured external output
    external.RI = RI_mat;
    external.ARI = ARI_mat;
    external.RI_mean = mean(RI_mat);
    external.ARI_mean = mean(ARI_mat);

    % Display summary
    fprintf('\n=== External Clustering Evaluation ===\n');
    fprintf('RI mean:  %.4f\n', external.RI_mean);
    fprintf('ARI mean: %.4f\n', external.ARI_mean);

    %-----------------------------
    % Internal clustering metrics
    %-----------------------------
    classification_mat = idx_mat;

    % Remove degenerate columns (single-label)
    singleCols = find( all(classification_mat == classification_mat(1,:), 1) );
    if ~isempty(singleCols)
        fprintf('Removing single-label columns: %s\n', mat2str(singleCols));
        classification_mat(:, singleCols) = [];
    end

    compactness_mat = zeros(1, size(classification_mat,2));
    dunnIndex_mat = zeros(1, size(classification_mat,2));
    overallSilhouette_mat = zeros(1, size(classification_mat,2));

    % Evaluate internal metrics
    for entry = 1:size(classification_mat,2)
        classification = classification_mat(:,entry);
        [labels, ~, ic] = unique(classification);
        K = numel(labels);

        %% Compactness
        centroids = arrayfun(@(i) mean(X(ic==i, :), 1), 1:K, 'UniformOutput', false);
        centroids = vertcat(centroids{:});
        sq_dists = sum((X - centroids(ic, :)).^2, 2);
        compactness_mat(entry) = sum(sq_dists);

        %% Dunn Index
        diam = zeros(K,1);
        for i = 1:K
            Xi = X(ic==i, :);
            if size(Xi,1) > 1
                D = pdist2(Xi, Xi);
                diam(i) = max(D(:));
            end
        end
        maxDiam = max(diam);

        minDist = inf;
        for i = 1:K-1
            Xi = X(ic==i, :);
            for j = i+1:K
                Xj = X(ic==j, :);
                D = pdist2(Xi, Xj);
                minDist = min(minDist, min(D(:)));
            end
        end
        dunnIndex_mat(entry) = minDist / maxDiam;

        %% Silhouette
        silhVals = silhouette(X, classification);
        overallSilhouette_mat(entry) = mean(silhVals);
    end

    % Package internal metrics
    internal.compactness = compactness_mat;
    internal.dunn = dunnIndex_mat;
    internal.silhouette = overallSilhouette_mat;
    internal.compactness_mean = mean(compactness_mat);
    internal.dunn_mean = mean(dunnIndex_mat);
    internal.silhouette_mean = mean(overallSilhouette_mat);

    fprintf('\n=== Internal Clustering Evaluation ===\n');
    fprintf('Compactness mean:       %.4f\n', internal.compactness_mean);
    fprintf('Dunn Index mean:        %.4f\n', internal.dunn_mean);
    fprintf('Overall Silhouette mean: %.4f\n', internal.silhouette_mean);

    %-----------------------------
    % Stability estimation
    %-----------------------------
    stability = HungarianMatching(classification_mat);
    fprintf('\n=== Stability Evaluation ===\n');
    fprintf('Stability score: %.4f\n', stability);
end


%==========================================================
% Subfunction: HungarianMatching
%----------------------------------------------------------
% Computes clustering stability using the Hungarian algorithm
% for optimal label alignment across multiple clustering runs.
%==========================================================
function [stability] = HungarianMatching(classification_mat)

    [~, R] = size(classification_mat);
    sumMatch = 0;

    for i = 1:R-1
        Ri = classification_mat(:,i);
        for j = i+1:R
            Rj = classification_mat(:,j);

            % Contingency matrix
            C = confusionmat(Ri, Rj);
            costMat = -C;
            costUnmatched = max(costMat(:)) + 1;
            [pairs, ~] = matchpairs(costMat, costUnmatched);

            % Label mapping
            f = zeros(max(Rj),1);
            for k = 1:size(pairs,1)
                f(pairs(k,2)) = pairs(k,1);
            end

            % Remap and accumulate matches
            Rj_mapped = f(Rj);
            sumMatch = sumMatch + mean(Ri == Rj_mapped);
        end
    end

    stability = (2/(R*(R-1))) * sumMatch;
end
