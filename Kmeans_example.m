% Example code for using k-means.  
% network parameters
k=10; % number of the input nodes
q=2; %number of output nodes = number of groups
m=3;
n=q+k+m; % size of the network. including Q, the total number of outputs
Q=q; % number of total outputs, full+empty
l=n+1;
gammain=ones(k,1);%dephasing rate for Vins%
gammadep=0;%dephasing rate%
gammaout=ones(q,1);%dephasing rate for Vouts%


%% Wave function's data set %%
num_states=50;
boundaries=[9 2;8 3;7 4; 6 5];
[phi,tags]=IPRgenerator(k,boundaries,num_states);

%% Cluster using K-means
opts = statset('MaxIter',1000);       % increase iterations if needed

% Perform K‑means (using 10 random starts for robustness)
[idx, C] = kmeans(phi, q, ...
    'Distance', 'sqeuclidean', ...
    'Replicates', 10, ...
    'Options', opts);
[RI_mat(b),ARI_mat(b)]=randomIndex(tags,idx);

