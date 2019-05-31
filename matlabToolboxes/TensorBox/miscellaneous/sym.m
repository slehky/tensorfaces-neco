function C = embedsym(X)
% Symmetric embedding of a tensor X into a symmetric tensor C
%

sz = size(X);nX = prod(sz);
n = numel(sz);

perm_a = perms(n:-1:1);
C = zeros(sum(sz)*ones(1,n));

sub_a = zeros(2,n);
csz = cumsum([0 sz]);
for kp = 1:numel(perm_a)
    kp
    perm_k = perm_a(kp,:);
    sz_p = sz(perm_k);
     
    for kn = 1:n
        sub_a(1:2,kn) = [csz(perm_k(kn))+1 csz(perm_k(kn)+1)];
    end
    
    str = '';
    for kn = 1:n
        str = [str sprintf('%d:%d,',sub_a(1,kn),sub_a(2,kn))];
    end
    str = sprintf('C(%s) = permute(X,perm_k);',str(1:end-1));
    
    eval(str);
end
