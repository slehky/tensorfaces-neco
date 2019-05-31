function Y = kron_unfolding3(Y,Ix_p,Jx_p)
% % Kronecker unfolding
% Y = A_1 \ox B_1 \ox C_1 + A_2 \ox B_2 \ox C_2 + ... + A_R \ox B_R \ox C_R
% Y: is a tensor, and K is array of size of patterns B_r
% This file is a part of the TENSORBOX.
N = ndims(Y);
SzY = size(Y);
Kx_p = SzY./(Ix_p.*Jx_p);
SzY_exp = [Kx_p; Jx_p; Ix_p]; SzY_exp = SzY_exp(:)';
Y = reshape(double(Y),SzY_exp);
Y = permute(Y,[3:3:3*N 2:3:3*N 1:3:3*N]);
Y = reshape(Y,prod(Ix_p),prod(Jx_p),[]); % a o b o c
