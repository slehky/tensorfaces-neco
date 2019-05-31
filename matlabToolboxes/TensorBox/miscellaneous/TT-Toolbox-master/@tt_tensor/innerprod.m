function y = innerprod(A,B)
% innerproduct between a TT-tensor and a tensor
% 
%
% Phan Anh-Huy, 2016

N = ndims(A);
if isa(B,'tt_tensor')
    Q = ttxtt(A,B,N,'left');
    y = reshape(core(A,N),1,[])*reshape(Q*reshape(core(B,N),rank(B,N),[]),[],1);
else
    Q = ttxt(A,B,N,'left');
    if isa(Q,'sptensor')
        y = innerprod(Q,tensor(core(A,N)));
    else
        y = reshape(core(A,N),1,[])*Q(:);
    end
end
