function [toep_id,P] = toeplitz_n(sz)
% sz: I1 x I2 x ... x 
% generate order- nToeplitz tensor of sequence 1:(I1+I2+...+In) - n+1
% 
%   P is linear operator which genrates a Toeplitz tensor T from a sequence t
%   i.e. 
%     vec(T) = P * t
% 
%   toep_id:  Toeplitz tensor of sequence 1,2,...,(I1+I2+...+In) - n+1.
%
% PHAN ANH-HUY.
%
n = numel(sz);
len = sum(sz)-n+1;
id = 1:len;

k = 1;
len_k = len;
toep_id = toeplitz(id(sz(1):-1:1),id(sz(1):len_k));

for k = 2:n-1
    len_k = len_k - sz(k-1)+1;
    toep_idk = toeplitz_n([sz(k) len_k-sz(k)+1]);
    
    toep_id = toep_id(:,toep_idk);
    toep_id = reshape(toep_id,size(toep_id,1)*sz(k),[]); 
end
toep_id = reshape(toep_id,sz);
P = speye(prod(sz),len);
P = P(toep_id(:),:);