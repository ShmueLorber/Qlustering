function [Cost,J]=CostFunction(H_mat,k,n,q,gammain,Lin2,Ldep,Lout,psi)
     %QuantumPropagation takes the properties of an open tight-binding network and
     %outputs the normalized current flowing from the output nodes in the steady-state using the Lindblad equation. 
     %notice sometimes modification of the Hamiltonian is neccesary
    l=n+1;%total number of sites%
    I=sparse(eye(l));
    J_vec=zeros(q,size(psi,1));
    psiWithGamaain=psi.*gammain'; % inserting gammain to the picture so we can lose this expression in the future

    for inx=1:size(psi,1)     %runs on each csi
        psihelp=zeros(1,l);
        psihelp(1,2:k+1)=psiWithGamaain(inx,:);
        psihelp=kron(conj(psihelp),psihelp);
        Lin1=zeros(l^2,l^2);
        Lin1(:,1)=psihelp;
        Lin=sparse(Lin1+sum(psi(inx,:)*psi(inx,:)')*Lin2);
        %final Lindblad equation 
        VonNeuman=-1i*(kron(I,H_mat)-kron(H_mat',I)); %commutation of H and rho%
        source=sparse(Lin); %dissipator for lindblad in operator%
        sink=Lout; %summing of vectorizations for Lout%
        L=VonNeuman+source+Ldep+sink; %full lindbladian%
        %finding rho of the steady state
        r=1e-8;  %making a regularization for more accurate results
        L=L+r*eye(size(L,1));
        [rho_vector,~]=eigs(L,1,'smallestabs');%finding the vector form of rho%
%         rho_vector=null(full(L));  %more acurate but slower kernel calculation
        %assuring trace=1
        diag_index=1:l+1:l^2;
        trace_of_rho=sum(rho_vector(diag_index,:));
        rho_vector_normalized=bsxfun(@rdivide,rho_vector,trace_of_rho);
        rho=reshape(rho_vector_normalized,l,l,size(rho_vector_normalized,2));
        %assuring real solutions%
        helper=rho_vector_normalized(diag_index,:);
        helper=round(helper,9); %sensitivity level for imaginary numbers
        i=helper==real(helper);
        i=all(i);
        rho_physical=rho(:,:,i);
        if size(rho_physical,3)>1
            rho_physical=rho_physical(:,:,1);
        end
        rho_physical=squeeze(rho_physical);
        if isempty(rho_physical)  % in case no real rho was found (numerical issue)
            rho_physical=eye(l);
        end
        % current calculation:
        helper=diag(rho_physical);
        J_vec(:,inx)=helper(l-q+1:l);
        J_tot=sum(J_vec(:,inx));
        J_vec(:,inx)=J_vec(:,inx)./J_tot; %normalization of J_vec
    end
    J=J_vec;
    %% Cost Function.
    J_max=J==max(J);
    Cost=sum(vecnorm((J_max-J)).^2);
end