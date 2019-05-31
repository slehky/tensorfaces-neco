function [hankel_id,P] = hankel_n(sz)
% sz: I1 x I2 x ... x 
% generate order- n Hankel tensor of sequence 1:(I1+I2+...+In) - n+1
% 
%   P is linear operator which genrates a Hankel tensor T from a sequence t
%   i.e. 
%     vec(T) = P * t
% 
%   toep_id:  Hankel tensor of sequence 1,2,...,(I1+I2+...+In) - n+1.
%
% PHAN ANH-HUY.
%
n = numel(sz);
len = sum(sz)-n+1;
id = 1:len;

len_k = len;

% HANKEL(C,R) is a Hankel matrix whose first column is C and whose last row is R.
hankel_id = hankel(id(1:sz(1)),id(sz(1):len_k));

for k = 2:n-1
    len_k = len_k - sz(k-1)+1;
    hankel_idk = hankel_n([sz(k) len_k-sz(k)+1]);
    
    hankel_id = hankel_id(:,hankel_idk);
    hankel_id = reshape(hankel_id,size(hankel_id,1)*sz(k),[]); 
end
hankel_id = reshape(hankel_id,sz);
P = speye(prod(sz),len);
P = P(hankel_id(:),:);