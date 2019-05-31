function [A,R,trave_val] = minrank_eqbound(Q,bound,exacterrorbound)
% Solve the mini
%    min      rank(A)
%subject to   g(A) = trace(A'*Q*A)=bound
%             A'*A = I  (A: orthogonal matrix)
%
% PHAN ANH-HUY, August.1 2018
% 

if nargin < 3
    exacterrorbound = true;
end
% First solve optimization with inequality constraint
%    min rank(A)  subject to trace(A'*Q*A) >= bound
%
% This have a solution consisting of R principal eigenvectors of Q
% such that
%  s1 + s2+... + sR >= bound > s1 + s2+... + s(R-1)
%
%  Note that if g(A* = U(:,1:R)) > bound, any other orthogonal matrix of rank R on
%  different subspace of A* does not have a higher g(A) than the bound.
%
%  g(U(1:R)) >= bound >= g(U(I))
%
%  where I represents a set of R indices U_I = U(:,[1<=i1 < i2 < ... <iR])

if norm(Q,'fro')^2 < bound 
    A = [];
    trave_val = [];R =[];
    return
end


[U,s] = eig(Q);
s = diag(s);
[s,ix] = sort(s,'descend');
U = U(:,ix);

R = find(cumsum(s)>=bound,1,'first');

if isempty(R)
    A = [];
    trave_val = []; 
    return
end

if R== size(Q,1)
    A = U;
    trave_val = sum(s);
    return
end

if exacterrorbound == false
    return
end
% Solution of the optimization with the equality contraint should have rank
% R, but we need more than R principal eigenvectors to represent A.
%
% A = U(:,1:R+1) * V
%
% where V is a matrix of size (R+1) x R
%
% The problem comes to solve
%     trace(V'*diag(s(1:R+1)) * V) = bound
%
% A simpler method is to keep (R-1) principle components of U_R,
% i.e, A(:,2:R) = U(:,2:R),
% while the first component of A will be adjusted from U1 and
% U(R+1) to hold the bound
%
% A = [u1*x1+u_(R+1)*x2  u2, u3, ...,u_R]
%
% where x1^2+x2^2 = 1
%
% bound = trace(A'*Q* A)
%       = s2+...+sR + s1*x1^2+ s(R+1) * x2^2
%       = (s1-s(R+1))*x1^2 + s2+...+sR + s(R+1)
%
% ->   x1^2 = (bound - sum(s(2:R+1))/(s1-s(R+1))
%
x1 = sqrt(max(0,(bound - sum(s(2:R+1)))/(s(1)-s(R+1))));
x2 = sqrt(max(0,1-x1^2));

A = [U(:,1)*x1+U(:,R+1)*x2 U(:,2:R)];
trave_val = bound;