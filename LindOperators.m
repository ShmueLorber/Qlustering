function [Lin2,Ldep,Lout] = LindOperators(l,k,q,gammain,gammadep,gammaout)
% [Lin1,Lin2,Ldep,Lout] = LindOperators(l,k,q,gammain,gammadep,gammaout)
        I=eye(l);
        Vout=zeros(l,l,q);
        Vin=zeros(l,l,k);%empty array of k lxl for Vin matrices%
        %Vin formation:
        Vhelp=I;
        Vhelp=reshape(Vhelp,[1 l l]);
        Vin(:,1,:)=Vhelp(:,:,2:k+1);
%         Vin=sum(Vin,3);
%         Lin2=-0.5*gammain*(kron(I,Vin'*Vin)+kron((Vin'*Vin).',I)); %-0.5*sqrt(gammain)*(kron(I,Vin'*Vin)+kron((Vin'*Vin).',I));
%         Lin2=sparse(Lin2/k);
        gammainhelp=reshape(gammain,[1 1 k]);
        Lin2=half_diss3(Vin,l);
        Lin2=gammainhelp.*Lin2;
        Lin2=sparse(sum(Lin2,3)/k);
        %Vdep formation:
        if gammadep~=0 %Vdep is pretty big (it's 3D is as big as l-(k+q+2)) so trying to reduce computation time
            Vdep=bsxfun(@times,pagetranspose(Vhelp(:,:,k+2:l-q)),Vhelp(:,:,k+2:l-q));
            Ldep=sparse(gammadep.*sum(diss3(Vdep,l),3));%sparse(sqrt(gammadep).*sum(diss3(Vdep,l),3));
        else
            Ldep=sparse(zeros(l^2));
        end
        %Vout formation:
        Vhelp=flip(I);
        Vhelp=reshape(Vhelp,[1 l l]);
        Vout(1,:,:)=Vhelp(:,:,1:q);
        gammaouthelp=reshape(gammaout,[1 1 q]); %reshape(sqrt(gammaout),[1 1 q]);
        Lout=diss3(Vout,l);
        Lout=gammaouthelp.*Lout;
        Lout=sparse(sum(Lout,3));

end