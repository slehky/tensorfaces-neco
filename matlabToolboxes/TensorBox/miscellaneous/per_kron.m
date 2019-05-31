function [Pleft,Pright] = per_kron(sA,sB)
% A o B = Pleft (B o A) Pright
%
% TENSORBOX
% Phan Anh Huy, 2012
Pleft = per_vectrans(sA(1),sB(1));
Pright = per_vectrans(sB(2),sA(2));