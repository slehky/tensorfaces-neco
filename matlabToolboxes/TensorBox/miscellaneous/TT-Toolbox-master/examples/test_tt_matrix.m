cd('/Users/phananhhuy/Documents/Matlab/TensorNetworks/TT-Toolbox-master')
setup;
cd('/Users/phananhhuy/Documents/Matlab/TensorNetworks/ttemps_2014_04_16')
install

addpath('Users/phananhhuy/Documents/Matlab/Dekron/')

addpath('/Users/phananhhuy/Documents/Matlab/RayleighRitz')
addpath('/Users/phananhhuy/Documents/Matlab/RayleighRitz/RencangLi');
addpath('/Users/phananhhuy/Documents/Matlab/TensorNetworks/ttemps_2014_04_16/GEVD')
% This code finds an apprximate to the smallest eigenvalue of two TT-matrix
% A and B,  B is positive definite. 
%
% Phan Anh-Huy, Nov-2014.


%% Generate symmetric matrix and construct a sym TTeMPS object of low-rank approximation to the matrix
clear all

I = 4; % dimension of block 
Sz = [256 256]; % dimension of the matrix 
N = log(Sz(1))/log(I(1)); % No of dimensions

% Generate a TT-matrix which is apprimate to a symmetric matrix 
dimp = [1:N; N+1:2*N];dimp = dimp(:)';
A = randn(Sz); lda = rand(Sz(1),1); A = A*diag(lda)* A';
A = reshape(A,I*ones(1,2*N));
Ap = permute(A,dimp);
Ap = reshape(Ap,I^2*ones(1,N));

tol = 1e0;
Att = tt_tensor(Ap,tol);
Att = TT_to_TTeMPS(Att);

% TT -> TTeMPS
Ua = cell(N,1);
for n = 1:N
    Ua{n} = Att.U{n};
    Ua{n} = reshape(Ua{n},[Att.rank(n) I I Att.rank(n+1)]); % rk
end
A_tt = TTeMPS_op(Ua');

A = full(A_tt.TTeMPS_op_to_TT_matrix);
A = max(A,A');

% Generate the second TT-matrix B which approximates a symmetric matrix

B = randn(Sz); lda = rand(Sz(1),1); B = B*diag(lda)* B';
B = reshape(B,I*ones(1,2*N));
Bp = permute(B,dimp);
Bp = reshape(Bp,I^2*ones(1,N));

% TT approximation
tol = 1e0;
Att = tt_tensor(Bp,tol);
Att = TT_to_TTeMPS(Att);

% TT -> TTeMPS
Ua = cell(N,1);
for n = 1:N
    Ua{n} = Att.U{n};
    Ua{n} = reshape(Ua{n},[Att.rank(n) I I Att.rank(n+1)]); % rk
end
B_tt = TTeMPS_op(Ua');

% Convert Block-TT tensor back to array matrix
B = full(B_tt.TTeMPS_op_to_TT_matrix);
B = max(B,B');

% Check positive definite B 
if min(eig(B)) > 0
    fprintf('OK! B is positive definite.\n')
end

% Now we have two TT-matrices A_tt and B_tt which are apprixate to two
% symetric matrices A and B. B is positive definitve.

%%
Attmat = TTeMPS_op_to_TT_matrix(A_tt);
Bttmat = TTeMPS_op_to_TT_matrix(B_tt);


%%
Cttmat = Attmat * Attmat;

Cttmat = round(Cttmat,1e-8);
