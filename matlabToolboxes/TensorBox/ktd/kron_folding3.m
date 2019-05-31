function Y = kron_folding3(Ykr,Ix,Jx,Kx)
% Kronecker folding
% 
% This file is a part of the TENSORBOX.
N = numel(Ix);
Y = reshape(Ykr,[Ix Jx Kx]);
Y = ipermute(Y,[3:3:3*N 2:3:3*N 1:3:3*N]);
Y = reshape(Y,Ix.*Jx.*Kx);
end