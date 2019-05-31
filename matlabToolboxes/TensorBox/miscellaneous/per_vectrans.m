function [P,Pidx] = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
%  vectorize of transposition
% TENSORBOX
% Phan Anh Huy, 2011
Pidx = reshape(1:m*n,[],n)'; Pidx= Pidx(:);
% if nargout ==2
    P = speye(m*n); P = P(Pidx,:);
% end
