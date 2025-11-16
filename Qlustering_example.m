% Usage example for Qlustering 
% k- # of input nodes
% m- # of intermediate nodes
% q- # of groups= # of output nodes
% num_states- size of the data set
% phi- data to cluster
% tags- predetermined labels
% centers- base points for groups initialization
% w- width of each group

k=3;
m=2;
q=4;

num_states=60;
centers = [0 10 0; 10 0 0; 0 0 10; -15 15 15; 0 10 -15]; % Centers for the 5 groups
width = 0.1;  
[phi,tags] = OverlapWaveFunctionGenerator(k,q,num_states,w,centers);

%% training parameters
numParticles=30;
it=70;


iterations=10; % Qluster 10 times for evaluation and consensus clustering
classification_mat=zeros(num_states,iterations);
RI_mat=zeros(1,iterations);
ARI_mat=zeros(1,iterations);
for run=1:iterations
    disp(run);
    [Hfinal,Jfinal,classification,RI,ARI] = Qlustering(k+q+m,k,q,numParticles,it,phi,tags);
    disp("Random index:")
    disp(RI)
    disp("Adjusted random index:")
    disp(ARI)
    classification_mat(:,run)=classification;
    RI_mat(:,run)=RI;
    ARI_mat(:,run)=ARI;
end
mean_RI=mean(RI_mat);
mean_ARI=mean(ARI_mat);
disp(mean_RI);
disp(mean_ARI);

% Second clustering using Hirarchical clustering of the consensus matrix
clusters_mat=ConsensusClustering(classification_mat,q);
[RI_Consensus,ARI_Consensus]=randomIndex(tags,clusters_mat);

disp("Consensus Random index:")
disp(RI_Consensus)
disp("Consensus Adjusted random index:")

disp(ARI_Consensus)
