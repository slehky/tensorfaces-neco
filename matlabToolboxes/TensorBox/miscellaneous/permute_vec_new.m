function P = permute_vec(In,n)
% vec(X_n) = P * vec(X)
Tt = reshape(1:prod(In),In);N = numel(In);
%Tn = tenmat(Tt,n);
Tn = permute(Tt,[n 1:n-1 n+1:N]);
Perm = Tn(:);
P = speye(prod(In));
P(:,Perm) = P;