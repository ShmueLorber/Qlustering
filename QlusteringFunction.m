function [H_trained, BestCostHistory]=QlusteringFunction(CostFunction,n,k,q,it,numParticles,Hmax,Hmin,tags)
    %%initialization
    BestCostHistory=zeros(3,it+1);
    GlobalBest.GroupSize = zeros(it,q);%q);
    minGroup=0; % size of the smallest group. preventing empty groups
    while minGroup==0
        H_initial=Hmin+(Hmax-Hmin)*rand(1,n^2); %Hamiltonian of the network
        %% 4 layers architecture. for actual Layer Indexed Matrix unindent the rows bellow
        OutMinus1hopping=1;   %hopping value for the out-1 layer
        %         H_mat=H_OutMinus2(H_initial,n,k,q,LayerIndexMatrix,OutMinus1hopping);
        H_mat=H_matrix(H_initial,n,k,q);
        %%
        %         if exist('LayerIndexMatrix','var')
        %             H_mat=H_matrix(H_initial,n,k,q,LayerIndexMatrix);
        %         else
%             H_mat=H_matrix(H_initial,n,k,q);
%         end
        % H_mat=sparse(H_matrix(H,n,k,q));
        GlobalBest.Position=H_mat;
%         disp(full(H_mat));
        empty_particle.Position=H_mat; %position
        empty_particle.Cost=[]; %cost will contain cost function values of the position
        particle=repmat(empty_particle,numParticles,1);
        [GlobalBest.Cost,J]=CostFunction(H_mat);
        J_max=J==max(J);
        GroupSize=sum(J_max,2);
        minGroup=min(GroupSize);
        GlobalBest.GroupSize(1,:) = sum(J_max,2);
%         [~, classification] = max(J_max); %finding the unsuprvised tags
    end
    disp(GroupSize);
    J_max=J==max(J);
    [~, classification] = max(J_max);
    [RI,ARI]=randomIndex(tags,classification); %keep track on success rates
    disp(['Iteration  0' ': Best Cost = ' num2str(GlobalBest.Cost) ' random index = ' num2str(RI) ' adjusted random index = ' num2str(ARI)]); %displaying of the iteraion number and best cost
    BestCostHistory(1,1)=GlobalBest.Cost;
    BestCostHistory(2,1)=RI;
    BestCostHistory(3,1)=ARI;
    %% main loop
    % j_vector=zeros(it,1)
    for i=1:it
        % Select a random index
        [rows, cols] = find(GlobalBest.Position ~= 0 & GlobalBest.Position ~= OutMinus1hopping);
        nonZeroEntries = length(rows);
        random_Index = randi(nonZeroEntries);% Select a random index
        for j=1:numParticles
            particle(j).Position=updateSparseMatrix(particle(j).Position,random_Index,rows,cols,Hmax,Hmin); %updating one entry of H, the position.
            [particle(j).Cost,J]=CostFunction(particle(j).Position);
            %% Cost Function
            J_max=J==max(J);
            [~, classification] = max(J_max); %finding the unsuprvised tags
            GroupSize=sum(J_max,2);
            % % Prevent empty entries
            % uniqueEntries = unique(classification); % Get the unique entries in the vector
            % result = numel(uniqueEntries)~=1; 
            if particle(j).Cost<GlobalBest.Cost && min(GroupSize)>=minGroup %result % find the best hamiltonian % 
                %update Best
                GlobalBest.Position=particle(j).Position;
                GlobalBest.Cost=particle(j).Cost;
                % [~, ~, ~, ~,Accuracy]=recognitionRates(tags,J); %keep track on success rates
                [RI,ARI]=randomIndex(tags,classification); %keep track on success rates
                disp(GroupSize);
            end
        end
        Hnew=GlobalBest.Position;
        parfor j=1:numParticles %apply the best hamiltonian to all particles
            particle(j).Position=Hnew;
        end
        [~,J]=CostFunction(GlobalBest.Position);
        J_max=J==max(J);
        GlobalBest.GroupSize(i+1,:)=sum(J_max,2);
        disp(['Iteration ' num2str(i) ': Best Cost = ' num2str(GlobalBest.Cost) ' random index = ' num2str(RI) ' adjusted random index = ' num2str(ARI)]); %displaying of the iteraion number and best cost
        BestCostHistory(1,i+1)=GlobalBest.Cost;
        BestCostHistory(2,i+1)=RI;
        BestCostHistory(3,i+1)=ARI;
        if RI==1 % stop the training if it is perfect
            break
        end
    end

    %% plot of the iterations propagation
    figure;
    %plot(BestCost,'LineWidth',2);
    semilogy(BestCostHistory(1,:),'LineWidth',2);
    xlabel('Iteration');
    ylabel('Best Cost');
%     legend("train");
    grid off;
    %% plot of the iterations propagation
    figure;
    %plot(BestCost,'LineWidth',2);
    plot(BestCostHistory(2,:),'LineWidth',2); hold on
    plot(BestCostHistory(3,:),'LineWidth',2);
    xlabel('Iteration');
    ylabel('Best Cost');
    legend("Random index","Adjusted random index");
    grid on;
    
    H_trained=GlobalBest;
end

%%
function [updatedMatrix] = updateSparseMatrix(matrix, entryToChange,rows,cols,valMax, valMin) %choses random number to insert in entryToChange in matrix
    % Select a random number within the given constraints
    randomValue = (valMax - valMin) * rand() + valMin;
    
    % Update the selected entry in the sparse matrix
    matrix(rows(entryToChange), cols(entryToChange)) = randomValue;
    % Preserve Hermiticity by updating the corresponding entry
    % in the conjugate position and using the complex conjugate value
    matrix(cols(entryToChange), rows(entryToChange)) = conj(randomValue);
    
    % Return the updated sparse matrix
    updatedMatrix = matrix;
end