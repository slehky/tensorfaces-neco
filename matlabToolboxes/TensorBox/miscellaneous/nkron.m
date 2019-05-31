function C = nkron(A,B)
% Nway tensor product (Kronecker product for matrices)
% Copyright by Phan Anh Huy, 2011-09
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by Phan Anh Huy.
%
% This file is a part of the TENSORBOX, 2012.

N = max(ndims(A),ndims(B));
Ia = size(A);Ia(end+1:N) = 1;
Ib = size(B);Ib(end+1:N) = 1;
Ic = Ia.*Ib;

tC = A(:) * B(:)';
tC = reshape(tC,[Ia Ib]);
tC = ipermute(tC,[2:2:2*N 1:2:2*N-1]);
C = reshape(tC,Ic);