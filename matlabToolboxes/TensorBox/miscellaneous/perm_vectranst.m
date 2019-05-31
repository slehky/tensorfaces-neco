function [Perm,P] = perm_vectranst(m,n)
% vec(X_mn) = P vec(X^T)
%  vectorize of transposition
M = reshape(1:m*n,[],n);
Mt = M';Perm = Mt(:);
if nargout >1
    P = speye(m*n);%eye(numel(M));
    P(Perm,:) = P;
end
 
end