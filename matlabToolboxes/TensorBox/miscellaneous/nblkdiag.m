function C = nblkdiag(A,B,expand)
% Blockdiagonal of two tensors
% Copyright 2012
% Phan Anh Huy, 08/06/2012
%
if isempty(A), C = B; return;end
if isempty(B), C = A; return;end
if nargin < 3
    expand = 0;
end
ndA = ndims(A); ndB = ndims(B);
szA = size(A); szB = size(B);
nd = max(ndA,ndB);
szA(end+1:nd) = 1;
szB(end+1:nd) = 1;

if expand == 1
    szA(end+1) = 1;
    szB(end+1) = 1;
end

szC = szA + szB;

if ~isa(A,'sptensor')
    A = sptensor(A);
end

if ~isa(B,'sptensor')
    B = sptensor(B);
end

subA = A.subs; subB = B.subs;
if expand == 1
    subA = [subA ones(size(subA,1),1)];
    subB = [subB ones(size(subB,1),1)];
end
subB = bsxfun(@plus,subB,szA);
    

C = sptensor([subA; subB],[A.vals ; B.vals], szC);
