clc
clear
close all

% Load data
load fisheriris           % gives meas (150x4) and species (150x1 cell array)

X = meas;                 % features
X=[X(:,1) X(:,3) X(:,4)]
q=3;
true_labels = grp2idx(species);  % convert species to numeric 1..3


opts = statset('MaxIter',1000);       % increase iterations if needed
% Perform K‑means (using 10 random starts for robustness)
idx_mat=zeros(size(true_labels,1),10);
sumd_mat=zeros(q,10);
for i=1:10
    [idx, C, sumd] = kmeans(X, q, ...
        'Distance', 'sqeuclidean', ...
        'Replicates', 1, ...
        'Options', opts);

        idx_mat(:,i) = idx;
        sumd_mat(:,i) = sumd;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% external values

RI_mat=zeros(1,10);
ARI_mat=zeros(1,10);
for i=1:10
    [RI_mat(i),ARI_mat(i)]=randomIndex(true_labels,idx_mat(:,i));
end
disp('RI_mat:');
RI_mat
disp('ARI_mat:');
ARI_mat
disp('RI mean:');
mean(RI_mat)
disp('ARI mean:');
mean(ARI_mat)
% internal values

classification_mat=idx_mat;
% find columns where every entry equals the first entry of that column
singleCols = find( all(classification_mat == classification_mat(1,:), 1) );
if ~isempty(singleCols)
    fprintf('Removing single‐label columns: %s\n', mat2str(singleCols));
    classification_mat(:, singleCols) = [];
end

% choose parameters to use
% classification=classification_mat(:,1);
compactness_mat=zeros(1,size(classification_mat,2));
dunnIndex_mat=zeros(1,size(classification_mat,2));
overallSilhouette_mat=zeros(1,size(classification_mat,2));
for entry=1:size(classification_mat,2)
    classification=classification_mat(:,entry);
    % Map labels to 1…K
    [labels, ~, ic] = unique(classification);
    K = numel(labels);

    %% 1. Compactness (within‐cluster sum of squares)
    centroids = arrayfun(@(i) mean(X(ic==i, :), 1), 1:K, 'UniformOutput', false);
    centroids = vertcat(centroids{:});                       % K×features
    sq_dists   = sum((X - centroids(ic, :)).^2, 2);         % N×1
    compactness = sum(sq_dists);
    compactness_mat(entry)=compactness;

    %% 2. Dunn Validity Index
    % (a) Intra‐cluster diameters
    diam = zeros(K,1);
    for i = 1:K
        Xi = X(ic==i, :);
        if size(Xi,1) > 1
            D     = pdist2(Xi, Xi);                         % pairwise distances
            diam(i) = max(D(:));
        end
    end
    maxDiam = max(diam);

    % (b) Minimum inter‐cluster distance
    minDist = inf;
    for i = 1:K-1
        Xi = X(ic==i, :);
        for j = i+1:K
            Xj = X(ic==j, :);
            D   = pdist2(Xi, Xj);
            minDist = min(minDist, min(D(:)));
        end
    end

    dunnIndex = minDist / maxDiam;
    dunnIndex_mat(entry)=dunnIndex;

    %% 3. Silhouette: overall score + plot
    silhVals          = silhouette(X, classification);    % also draws the plot
    overallSilhouette = mean(silhVals);
    % figure;                                           % open a new figure window
    % [silhVals, h] = silhouette(X, classification); % compute and plot
    % xlabel('Silhouette Value');
    % ylabel('Cluster');
    % title('Silhouette Plot');
    % overallSilhouette = mean(silhVals);
    overallSilhouette_mat(entry)=overallSilhouette;
end
fprintf('compactness mean: %f\n', mean(compactness_mat));
fprintf('Dunn Index mean: %f\n',mean(dunnIndex_mat));
fprintf('Overall Silhouette mean: %f\n',mean(overallSilhouette_mat));

% stability
stability=HungarianMatching(classification_mat);

function [stability]=HungarianMatching(classification_mat)

    [N, R] = size(classification_mat);
    sumMatch = 0;

    for i = 1:R-1
        Ri = classification_mat(:,i);
        for j = i+1:R
            Rj = classification_mat(:,j);

            % contingency
            C = confusionmat(Ri, Rj);
            costMat      = -C;
            costUnmatched = max(costMat(:)) + 1;
            [pairs, ~] = matchpairs(costMat, costUnmatched);

            % build mapping f
            f = zeros(max(Rj),1);
            for k = 1:size(pairs,1)
                f(pairs(k,2)) = pairs(k,1);
            end

            % relabel and accumulate match fraction
            Rj_mapped  = f(Rj);
            sumMatch  = sumMatch + mean(Ri == Rj_mapped);
        end
    end

    stability = (2/(R*(R-1))) * sumMatch;
    fprintf('Stability (label‐aligned) over %d runs: %.4f\n', R, stability);
end