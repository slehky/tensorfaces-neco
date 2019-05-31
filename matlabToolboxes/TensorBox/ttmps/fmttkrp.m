function G = fmttkrp(X,A,B,mode_n)
%  X_(n)^T * khatrirao(B,A)
%
szX = size(X);
if numel(szX)<3, szX(3) = 1;end
R = size(A,2);
switch mode_n
    case 1
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[1 szX(2) R]));
        G = squeeze(sum(G,2));
    case 2
        tmp = reshape(X,[szX(1)*szX(2) szX(3)])*B;
        G = bsxfun(@times, reshape(tmp,[szX(1) szX(2) R]), reshape(A,[szX(1) 1 R]));
        G = reshape(sum(G,1),size(G,2),[]);
    case 3
        tmp = A.'*reshape(X,[szX(1)  szX(2)*szX(3)]);
        G = bsxfun(@times, reshape(tmp,[R szX(2) szX(3)]), B.');
        G = squeeze(sum(G,2)).';
end
    
end
