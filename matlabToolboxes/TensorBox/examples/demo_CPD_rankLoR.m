% This file illustrates decomposition of a tensor into multiple Kronecker
% product terms of rank-Lp and rank-Mp tensors Ap and Xp, for p = 1,...,P
%
%  Y = A1  \otimes X1 +  A2  \otimes X2 + ... + AP  \otimes XP
%
% \otimes: Kronecker product
%
% P: number of patterns or product terms
%
% Ap:  Jp1 x Jp2 x ... x JpN
% Xp:  Kp1 x Kp2 x ... x KpN

% Special case: Ap and Xp have singleton dimensions (i.e., Jpn = 1 or Kpn
% =1)
% Modes of tensors Ap are specified by modes_A
% Modes of tensors Xp are specified by modes_X = {1:N}\modes_A
%
% Phan Anh Huy, 2012
%
% Example 1 in the manuscript 
% Anh Huy Phan, Andrzej Cichocki, Petr Tichavsky, Rafal Zdunek and
% Sidney Lehky, FROM BASIS COMPONENTS TO COMPLEX STRUCTURAL PATTERNS,
% 2012.
% The codes need the tensor toolbox (ver 2.4).

%% DEMO rankLoR CPD
% Generate a synthectic tensor of size I1 x I2 x... x IN
clear all
In = [10 10 10 10 10 10];   % tensor size
N = numel(In);              % order of data tensor

% Modes of tensors A
modes_A = [1 2 3];                % the first three modes are explained by tensors A_p
modes_X = setdiff(1:N,modes_A);   % modes of tensors X_p (with some singleton modes)

% Ranks of tensors A and X
ranks_A = [2 2 3];% rank of patterns A_p, ranks_A = [L1, L2,..., LP]
ranks_X = [1 2 2];% rank of patterns X_p, ranks_X = [M1, M2,..., MP]

% Other parameters
ranks_A_X = [ranks_A;ranks_X];
NdimA = numel(modes_A);    % order of tensor A_p
NdimX = numel(modes_X);    % order of tensor X_p
P = numel(ranks_A); % number of Kronecker terms

% Factor matrices
U = cell(numel(modes_A),P); % U{1,p}, U{2,p}, ..., U{end,p}: factor matrices of tensor Ap
V = cell(numel(modes_X),P); % V{1,p}, V{2,p}, ..., V{end,p}: factor matrices of tensor Xp

%% Generate Y form P Kronecker products
Y = 0;
for p = 1:P
    % tensor Ap
    U(:,p) = arrayfun(@(x) randn(x,ranks_A(p)),In(modes_A),'uni',0);
    % tensor Xp
    V(:,p) = arrayfun(@(x) randn(x,ranks_X(p)),In(modes_X),'uni',0);
    
    Ap = full(ktensor(U(:,p)));
    Xp = full(ktensor(V(:,p)));
    % correct dimensions of Xp with NdimA singleton modes
    Xp = reshape(Xp,[ones(1,NdimA) In(modes_X)]);
    Y = Y + nkron(full(Ap),Xp);                      % KTD model Eq. (1) 
end
Y = tensor(Y);
SNR = 20; % Noise level : inf dB for noiseless tensor
%               30 dB for SNR = 30 dB
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(In));
if ~isinf(SNR)
    Y = full(Y);
    Y = Y + sig2 * randn(In);
end
Y = tensor(Y);

%% Factor matrices of the tensor Y
QL = [];
QM = [];
for p = 1:P
    QL = blkdiag(QL,kron(eye(ranks_A_X(1,p)),ones(1,ranks_A_X(2,p))));  % Eq. (7)
    QM = blkdiag(QM,kron(ones(1,ranks_A_X(1,p)),eye(ranks_A_X(2,p))));  % Eq. (8)  
end

Wtilde = cell(N,1);
W = cell(N,1);

for n = 1:NdimA
    Wtilde{modes_A(n)} = cell2mat(U(n,:));               % Eq. (5) 
    W{modes_A(n)} = Wtilde{modes_A(n)} * QL;            % Eq. (4)   
end
for n = 1:NdimX
    Wtilde{modes_X(n)} = cell2mat(V(n,:));               % Eq. (6) 
    W{modes_X(n)} = Wtilde{modes_X(n)} * QM;            % Eq. (4) 
end

%% Cramer round induced bound
crib = cribBCDLoR(Wtilde,ranks_A_X,modes_X);
crib = sig2^2*crib;
for n =1:N
    fprintf('CRIB %s dB \n',sprintf('%.2f ',-10*log10(crib(n,1:size(Wtilde{n},2)))))
end

fprintf('Mean CRIB %.2f dB \n',-10*log10(nanmean(crib(:))))

%% Decomposition: may try several times to obtain a good estimate
opts = bcdLoR_als;
opts.MaxRecursivelevel = 3;
opts.init = 'random';
opts.alsinit = 0;
opts.maxiters= 1000;
opts.printitn = 1;
[Wthat,output,What] = bcdLoR_als(Y,[ranks_A;ranks_X],modes_X,opts);   % Eqs. (9) (10)

%% Square angular error between estimated and original factor components
[msae,foe,sae1] = SAE(Wtilde,Wthat);
fprintf('\n');
fprintf('Squared angular errors for components of tensor Ap\n');

for n = modes_A
    fprintf('Factor %d (%dx%d)  %s dB \n',n,In(n),sum(ranks_A),...
        sprintf('%.2f ',-10*log10(sae1(n,1:size(Wtilde{n},2)))))
end
fprintf('\n');
fprintf('Squared angular errors for components of tensor Xp\n');
for n = modes_X
    fprintf('Factor %d (%dx%d)  %s dB \n',n,In(n),sum(ranks_X),...
        sprintf('%.2f ',-10*log10(sae1(n,1:size(Wtilde{n},2)))))
end

fprintf('Mean SAE %.2f dB \n',-10*log10(nanmean(sae1(:))))
fprintf('Mean CRIB %.2f dB \n',-10*log10(nanmean(crib(:))))

fprintf('High SAE (dB) (~~CRIB) means high accuracy\n');

%% Visualize approximation error

figure(1); set(gca,'fontsize',18)
semilogy(1-real(output.Fit))
xlabel('Iteration')
ylabel('Relative Approximation Error')

% End of example

%% Nonnegative constraints
% opts = bcdLoR_als;
% opts.MaxRecursivelevel = 3;
% opts.init = 'random';
% opts.alsinit = 0;
% opts.maxiters= 10000;
% opts.printitn = 1;
% [Uhat,output,Ahat] = nbcdLoR_mls(Y,[ranks_A;ranks_X],modes_X,opts);
