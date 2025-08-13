function[u3]=half_diss3(A,l)
% 3D version of the second part of the lindblad superoperator (-0.5*{,})
I=repmat(eye(l),[1 1 size(A,3)]);
u3=-0.5.*(ThreeDkron(I,pagemtimes(pagetranspose(conj(A)),A))+ThreeDkron(pagetranspose(pagemtimes(conj(pagetranspose(A)),A)),I));

end