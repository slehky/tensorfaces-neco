function Y = kron_foldingN(Ykr,patch_sizes)
% Kronecker folding
% 
% This file is a part of the TENSORBOX.
N = numel(patch_sizes{1}); % order of the tensor 

patch_sizes = cell2mat(patch_sizes(:));

No_patches = size(patch_sizes,1);
perm_idx = [];
for k = No_patches:-1:1
    perm_idx = [perm_idx k:No_patches:No_patches*N];
end
kr_sizes= patch_sizes';
Y = reshape(Ykr,kr_sizes(:)');
Y = ipermute(Y,perm_idx);
Y = reshape(Y,prod(patch_sizes,1));
end