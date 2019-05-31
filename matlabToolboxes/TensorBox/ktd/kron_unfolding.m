function C = kron_unfolding(C,K)
% Kronecker unfolding
% C = A_1 \ox B_1 + A_2 \ox B_2 + ... + A_R \ox B_R
% C: is a tensor, and K is array of size of patterns B_r
% This file is a part of the TENSORBOX.
%
I = size(C);N = ndims(C);
J = I./K;
In = [K; J];In = In(:)';
C = reshape(C,In);
C = permute(C,[2:2:2*N 1:2:2*N-1]);
C = reshape(C,prod(J),[]);