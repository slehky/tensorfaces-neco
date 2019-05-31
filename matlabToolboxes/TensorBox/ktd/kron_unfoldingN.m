function Y = kron_unfoldingN(Y,patch_sizes)
% % Kronecker unfolding
% Y = A_1 \ox A_2 \ox ... \ox A_N
% Y: is a tensor, and patch_sizes{n} indicates of patterns A_n
% This file is a part of the TENSORBOX.
N = ndims(Y);
SzY = size(Y);
% Check and correct size of the last patterns A_N
psz = patch_sizes(1:end-1);
patch_sizes{end} = SzY./prod(cell2mat(psz(:)),1);
patch_sizes = cell2mat(patch_sizes(:));

SzY_exp = patch_sizes(end:-1:1,:); SzY_exp = SzY_exp(:)';
Y = reshape(double(Y),SzY_exp);
No_patches = size(patch_sizes,1);
perm_idx = [];
for k = No_patches:-1:1
    perm_idx = [perm_idx k:No_patches:No_patches*N];
end
Y = permute(Y,perm_idx);
unfolding_size = prod(patch_sizes,2);
%unfolding_size = prod(patch_sizes(end:-1:1,:),2);
Y = reshape(Y,unfolding_size'); % a o b o c
%Y = permute(Y,[ndims(Y):-1:1]);