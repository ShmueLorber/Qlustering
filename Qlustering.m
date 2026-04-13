function [Hfinal,Jfinal,classification,RI,ARI] = Qlustering(n,k,q,numParticles,it, phi,tags)
    %This code uses the quantum transport to make quantum clustering, or
    %Qlustering. 
    % n- the size of the network
    % k- # of input nodes
    % q- # of classes
    % phi- data to cluster
    % tags- data external labels. Used only for evaluation
    % it- number of iterations
    % numParticles- number of parallel Hamiltonians used in each iteration

    l=n+1;
    gammain=ones(k,1);%dephasing rate for Vins%
    gammadep=0;%dephasing rate%
    gammaout=ones(q,1);%dephasing rate for Vouts%
    
    % problem parameters
    % it=100; %number of iterations. can be replaced with a function
    % numParticles=20;
    Hmax=200;
    Hmin=-200;
    
    %% training
    [Lin2,Ldep,Lout]=LindOperators(l,k,q,gammain,gammadep,gammaout);   % Lindblad superoperators. does not depand on H
    CostFunction=@(H)QmeansCost2(H,k,n,q,gammain,Lin2,Ldep,Lout,phi);   % Lindblad equation as a cost function
    H=QlusteringFunction(CostFunction,n,k,q,it,numParticles,Hmax,Hmin,tags); % Qlustering training process
    
    %% final evaluation:
    Hfinal=H.Position;
    Jfinal=QuantumPropagation(Hfinal,k,n,q,gammain,Lin2,Ldep,Lout,phi); % Lindblad equation
    %%
    J=Jfinal; %classification current
    %% Cost Function.
    J_max=J==max(J);
    %%
    [~, classification] = max(J_max);
    disp(classification);
    [RI,ARI]=randomIndex(tags,classification);
    disp(RI);
    disp(ARI);
    sum(J_max,2)
end
