function clusters = ConsensusClustering(classification_mat,q)
    % UPGMA clustering on a 60×60 consensus matrix
    % consensus_mat: co‐association matrix with values in [0,1]
    [num_states, num_runs] = size(classification_mat);
    
    % --- 1) Build consensus (co‑association) matrix
    consensus_mat = zeros(num_states);
    for i = 1:num_states
        for j = i:num_states
            % fraction of runs where i and j share the same label
            consensus_mat(i,j) = mean(classification_mat(i,:) == classification_mat(j,:));
            consensus_mat(j,i) = consensus_mat(i,j);
        end
    end
    % --- 2) Convert similarity to distance
    distMat = 1 - consensus_mat;
    
    % --- 3) Convert to condensed distance vector
    % Requires Statistics and Machine Learning Toolbox
    distVec = squareform(distMat);
    
    % --- 4) Compute hierarchical clustering (average linkage = UPGMA)
    Z = linkage(distVec, 'average');
    
    % --- 5) (Optional) Plot dendrogram
    figure;
    dendrogram(Z);
    title('UPGMA Dendrogram');
    xlabel('Data Point Index');
    ylabel('Distance');
    
    % --- 6) Assign each of the 60 points to one of numClusters clusters
    clusters = cluster(Z, 'maxclust', q);
    
    % --- 7) Display or return assignments
    disp('Cluster assignments (1×60):');
    disp(clusters');
end