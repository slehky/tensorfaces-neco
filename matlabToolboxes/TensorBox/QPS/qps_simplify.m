function [snew,cnew,A] = qps_simplify(s,c)
% Simplify the QP over sphere when s have identical coefficients
%   min 1/2 x'* diag(s) * x + c' * x
%   subject to x'*x = 1
%
% Phan Anh Huy, 2017
%  snew:  s1<s2 <... < sL
%  cnew:  cnew(1) = norm([c_k1, ...ckl]:  sk1 = ... = skl]
%  c = A * cnew
%   
% s(1) == 1
thresh_ident = 1e-10;
sd = diff([0 ;s(:)+1]);% plus s+1, to deal with the case when s(1) is close to zero, 
mask = abs(sd)>thresh_ident;
snew = s(mask);
cid = find(mask==1);
cid = [cid ; numel(c)+1];
cnew = zeros(numel(cid)-1,1);

A = zeros(numel(c), numel(cnew));
for k = 1: numel(cid)-1
    id1 = cid(k);
    id2 = cid(k+1)-1; 
    if id2 > id1
        cnew(k) = norm(c(id1:id2));
    else
        cnew(k) = c(id1);
    end
    A(id1:id2,k) = c(id1:id2)./cnew(k);
end
