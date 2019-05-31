function Y = dbldiagonal(A)
% Generate diagonal tensor from A
% If A is a vector, Y is diagonal matrix
% If A is a matrix, Y is 3-D tensor whose diagonal slice is A
%
% A: N-D tensor  -> Y: (N+1)-D sparse diagonal tensor
%
% Copyright 2012
% Phan Anh Huy, 08/06/2012

szA = size(A);
szAex = reshape([szA;szA],1,[]);
tt = [1 cumprod(szAex)];
Y = zeros(szAex);
for i = 1:prod(szA)
    sbidx = ind2sub_full(szA,i);
    sbidxex = reshape([sbidx;sbidx],[],1);
    sbidxex(2:end) = sbidxex(2:end)-1;
    idx2 = tt(1:end-1) * sbidxex;
    Y(idx2) = A(i);
end