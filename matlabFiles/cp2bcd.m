function V = cp2bcd(U,LR,rankRdim)
% BCD components to BCD tensor
% BCD = Block Canonical Polyadic Decomposition
% P: dimensions of the pattern tensor A
% Lr: ranks of A. Number of elements of Lr: no. patterns 
% Phan Anh Huy, 2012

In = cellfun(@(x) size(x,1),U);
if size(LR,1) == 1
    LR(2,:) = 1;
end

N = numel(In);
rankLdim = setdiff(1:N,rankRdim);
V = cell(numel(rankLdim),size(LR,2));
for p = rankLdim
    V(p,:) = mat2cell(U{p},In(p),LR(1,:));
end
for p = rankRdim
    V(p,:) = mat2cell(U{p},In(p),LR(2,:));
end
