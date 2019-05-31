function [X,infos] = lr_mat(Y,R)

[u,s,v] = svd(Y);
X = u(:,1:R)*s(1:R,1:R)*v(:,1:R)';
infos = struct;