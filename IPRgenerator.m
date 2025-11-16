function [phi,tags]=IPRgenerator(k,boundaries,num_states)
    %==========================================================
    % Generates a dataset of state vectors (phi) with controlled localization
    % characterized by their Inverse Participation Ratio (IPR).
    % Each half of the dataset has IPR values confined to different ranges.
    %----------------------------------------------------------
    % INPUTS:
    %   k           - Dimension of each state vector
    %   boundaries  - Two-element vector [low, high] defining target IPR limits
    %   num_states  - Total number of states to generate
    %
    % OUTPUTS:
    %   phi   - Matrix of generated normalized state vectors (num_states x k)
    %   tags  - Class labels (1 or 2) based on IPR threshold
    %==========================================================

    % Preallocate result arrays for efficiency
    RI_mat=zeros(1,size(boundaries,1));
    ARI_mat=zeros(1,size(boundaries,1));
    phi = zeros(num_states,k);
    ipr=zeros(num_states,1);

    % Initialize IPR ranges for both groups
    ipr(1:num_states/2)=boundaries(2);      % First group starts above upper bound
    ipr((num_states+1)/2:num_states)=boundaries(1); % Second group starts below lower bound

    % Parallel loop to generate each state vector
    parfor i = 1:num_states
        if i<=num_states/2
            % ---- Group 1: Generate more delocalized states (lower IPR) ----
            while ipr(i)<boundaries(1)
                phi(i,:) = 0.2+0.4*rand(k,1);                % Random amplitudes in [0.2, 0.6]
                phi(i,:) = sqrt(phi(i,:).^2/sum(phi(i,:).^2)); % Normalize to unit norm
                SUM4 = phi(i,:).^4;                          % Compute fourth power of amplitudes
                ipr(i) = 1/sum(SUM4);                        % Compute IPR = 1 / sum(|phi|^4)
            end
        elseif i>num_states/2
            % ---- Group 2: Generate more localized states (higher IPR) ----
            while ipr(i)>boundaries(2)
                phi(i,:) = 0.6.*rand(k,1);                   % Random amplitudes in [0, 0.6]
                phi(i,:) = sqrt(phi(i,:).^2/sum(phi(i,:).^2)); % Normalize to unit norm
                SUM4 = phi(i,:).^4;
                ipr(i) = 1/sum(SUM4);
            end
        end
    end

    %----------------------------------------------------------
    % Define special edge states to serve as boundary examples
    %----------------------------------------------------------
    phi(1,:)=sqrt(1/k).*ones(1,k);           % Fully delocalized uniform state
    phi(2,:)=sqrt(1/(k-1)).*ones(1,k);       % Almost uniform state with one zero
    phi(2,1)=0;
    phi(num_states,:)=zeros(1,k);            % Fully localized state
    phi(num_states,1)=1;

    %----------------------------------------------------------
    % Tagging states into two groups based on their IPR
    %----------------------------------------------------------
    tags=ipr>boundaries(1);                  % Logical separation by threshold
    tags=double(tags+1);                     % Convert to numerical class labels (1,2)
end