function consensus_mat = consensus_matrix(clustering_mat)
    % The function assign group to each data point based on initial number
    % of groups, q, and clustering_mat, a matrix of various clusterings of
    % the data. The key principles ere are:
    % 1. There are only q groups
    % 2. Dont create new connections between data points (that the cluster algorithm didnt already created)
    % 3. Erase as minimum number of connections as you can
    % Parameters
    [num_states, num_iter] = size(clustering_mat);
    % Initialize consensus matrix
    consensus_mat = zeros(num_states);
    
    % Build consensus matrix efficiently
    for iter = 1:num_iter
        % Create a co-occurrence matrix for the current iteration
        curr_labels = clustering_mat(:, iter);
        co_occurrence = curr_labels == curr_labels';
        
        % Accumulate the co-occurrences into the consensus matrix
        consensus_mat = consensus_mat + co_occurrence;
    end
    
    % Normalize the consensus matrix
    consensus_mat = consensus_mat / num_iter;
end
