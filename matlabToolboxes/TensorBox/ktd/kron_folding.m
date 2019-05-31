function Y = kron_folding(Ym,Ix,Ia)
% Kronecker folding
% 
% This file is a part of the TENSORBOX.
N = numel(Ix);
Y = reshape(Ym,[Ia Ix]);
Y = ipermute(Y,[2:2:2*N 1:2:2*N]);
Y = reshape(Y,Ix.*Ia);
end