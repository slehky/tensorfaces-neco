function [A_mps,A_tt] = double2tt_matrix(A,Sz1,Sz2,tol)
% Convert a matrix A to a TT-matrix A with a specific tolerance tol

% Nodes of A have size of Rk(n) x Sz1(n) x Sz2(n) x Rk(n+1)
% 

%%
N = numel(Sz1); % number of dimension of TT-matrix A 
dimp = [1:N; N+1:2*N];dimp = dimp(:)';

A = reshape(A,[Sz1(:) ; Sz2(:)]');
Ap = permute(A,dimp);
Ap = reshape(Ap,Sz1.*Sz2);

% TT approximation
A_tt = tt_tensor(Ap,tol);
A_tt = TT_to_TTeMPS(A_tt);

% TT -> TTeMPS
Ua = cell(1,N);
for n = 1:N
    Ua{n} = A_tt.U{n};
    Ua{n} = reshape(Ua{n},[A_tt.rank(n) Sz1(n) Sz2(n) A_tt.rank(n+1)]); % rk
end
A_mps = TTeMPS_op(Ua);
if nargin >=2 
    A_tt = A_mps.TTeMPS_op_to_TT_matrix;
end

end