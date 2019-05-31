function sdx = ind2sub_full(siz,ndx)
%IND2SUB Multiple subscripts from linear index.
% Modify the Matlab function IND2SUB
% Anh Huy Phan 2008
siz = double(siz);
sdx = zeros(numel(ndx),numel(siz));

n = length(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi = rem(ndx-1, k(i)) + 1;
    vj = (ndx - vi)/k(i) + 1;
    sdx(:,i) = vj;
    ndx = vi;
end
