function X = tt_prod(A,B)
% TT-product: contraction along last-mode of A nd first-mode of B
%
%
% TENSORBOX, 2018
szA = size(A);
szB = size(B);
if szA(end) == szB(1)
    X = reshape(A,[],szA(end)) * reshape(B,szB(1),[]);
    X = reshape(X,[szA(1:end-1) szB(2:end)]);
end
end