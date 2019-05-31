function ftoeplitz = hankelization(sz)
% Generate Hankelization operator and its ajdoint operator for the size "sz"
%
% PHAN ANH-HUY, 2016
%

persistent Hankel_idx sz_ dPP P;

if isempty(sz_) || (numel(sz_) ~= numel(sz)) || any(sz ~= sz_)
    [Hankel_idx,P] = hankel_n(sz);
    dPP = diag(P'*P); % scaling factor
    sz_ = sz;
end

ftoeplitz.A = @(x)  reshape(x(Hankel_idx(:),:),[sz(:)' size(x,2)]);
ftoeplitz.At = @(x)  bsxfun(@rdivide,P'*reshape(x,prod(sz),[]),dPP);



