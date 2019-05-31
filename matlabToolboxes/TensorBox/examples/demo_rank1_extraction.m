% This example illustrates extraction of a rank-1 tensor from a rank-R
% tensor. 
% 
% The code uses the ASU algorithm.
% 
% TENSOR BOX, v1. 2014
% 
%% Generate a tensor of size 30 x 30 x 30 and rank 4

I = 30; %tensor size 
N = 3; % tensor order
R = 4; % tensor rank

A = cell(N,1);
for n = 1:N
    A{n} = randn(I,R);
    A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));
end
lambda = ones(R,1);
Y = tensor(ktensor(lambda,A));

% Add Gaussian noise to the tensor Y
SNR = 10; % Noise level in dB, inf for noiseless tensor
sigma_noise = 10^(-SNR/20)*std(double(Y(:)));
Y = Y + sigma_noise * randn(size(Y));
 
% Compute Cramer-Rao Induced Bound for estimating A
cribA = cribCP(A,[],sigma_noise^2); 
 
%% Set parameters for the ASU algorithm
opts = bcdLp1_asu3;  % Get default parameters 
opts.tol = 1e-6;
opts.maxiters = 20;
opts.alsinit = 0;       
opts.refine_a = true;
opts.printitn = 1;
opts.init = 'random'; % or other initialization e.g., 'ceig', 'dtld' or 'nvec'


% tic;
[ah,Uh,lambda,cost] = bcdLp1_asu3(Y,R,opts);
% t_rk1 = toc;

%% Compute squared angular error and compare it with its Cramer-Rao induced bound

[mase_1,mase_,sae] = SAE(ah,A);
[foe,compid] = min(min(sae,[],1));


fprintf('Mean squared angular error %.3f dB \n',-10*log10(mase_));
fprintf('Cramer-Rao Induced bound on the estimated factors %2.2f dB\n',-10*log10(mean(cribA(:,compid))))

% Plot the cost value vs iteration
figure(1); clf;
semilogy(cost(:,end));
xlabel('iterations')
ylabel('Approximation Error')
