function C = kronl(A,B)
%
sA = size(A);
sB = size(B);
C = kron(A,B);
C = reshape(C,[sB(1) sA(1) sB(2) sA(2)]);
C = permute(C,[2 1 4 3]);
C = reshape(C,sA.*sB);