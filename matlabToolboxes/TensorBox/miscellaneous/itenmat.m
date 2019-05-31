function At = itenmat(A,mode,I)
N = numel(I);
ix = [mode setdiff(1:N,mode)];
At = reshape(A,I(ix));
At = permute(At,[2:mode 1 mode+1:N]);