function [u,s,v,approx_error] = lowrank_matrix_approx(T,error_bound,exacterrorbound)
% find the best low-rank approximation to T
% such that |T-Tx|_F^2 <= error_bound
%
%%
% TENSORBOX, 2018
if nargin <3
    exacterrorbound = true;
end
[u,s,v]=svd(T,'econ');
s=diag(s);
cs = cumsum(s.^2);
r1 = find((cs(end) - cs) <= error_bound,1,'first');
u=u(:,1:r1); s=s(1:r1);v = v(:,1:r1);
approx_error = cs(end)-cs(r1);% |T-Tx|_F^2

if exacterrorbound    
    s(1) = s(1)+sqrt(error_bound-approx_error); %s(1)-sqrt(error_bound-approx_error); % 
    approx_error = error_bound;
end