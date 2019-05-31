% This example compares th TT-2-CPD conversion with CP-ALS
% in decomposition of tensor with highly collinear factors
%
%% Generate a tensor
clear all;

% Tensor of order-5 and size 10 x 10 x 10 x 10 x 10
N = 5; % tensor order
I = 10*ones(1,N); % tensor size
R = 10;  % tensor rank

% Random factor matrices have collinearity coefficients distributed in specific ranges
c = [0.97 .97; 0.95 0.99; 0.9 0.99 ; 0.9 0.99; 0.9 0.999;0.9 0.99; 0.9 0.999];

A = cell(N,1);
for n = 1:N
    if R<=I(n)
        A{n} = gen_matrix(I(n),R,c(n,:));
    else
        A{n} = randn(I(n),R);
    end
    A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));
end

lambda = ones(R,1);
Y = tensor(ktensor(lambda,A));

% Add Gaussian noise to tensor Y
SNR = 20; % Noise level in dB, inf for noiseless tensor
sigma_noise = 10^(-SNR/20)*std(double(Y(:)));
Y = Y + sigma_noise * randn(size(Y));

% Cramer-Rao induced bound on estimation of factor matrices
crib_A = cribCP(A,[],sigma_noise^2);

%% Algorithm 1:  ALS TT_conversion algorithm
opts = cp_ttconv;
opts.tol = 1e-8;
opts.maxiters = 2000;
[P,output] = cp_ttconv(Y,R,opts);

[msae1,msae2,sae_a,sae2] = SAE(A,P.u);
sae_ttals = msae1;

fprintf('CRIB      %.2f dB\n',mean(-10*log10(crib_A(:))));
fprintf('MSAE(ALS) %.2f dB\n',mean(-10*log10(sae_a(:))));

[msae1,msae2,sae_a,sae2] = SAE(output.Uinit,A);
fprintf('MSAE(init)%.2f dB\n',mean(-10*log10(sae_a(:))));
sae_ttexact = msae1;


%% CPD 
opts = cp_fastals;
opts.printitn = 1;
opts.tol = 1e-8;
opts.maxiters = 2000;
opts.linesearch = false;
opts.init = 'nvec';
[P,out2] = cp_fastals(Y,R,opts);

[msae1,msae2,sae_a,sae2] = SAE(A,P.U);

fprintf('CRIB %.2f\n',mean(-10*log10(crib_A(:))));
fprintf('MSAE %.2f\n',mean(-10*log10(sae_a(:))));
sae_cpals = msae1;

%%
clc
fprintf('            MSAE (dB)\n');
fprintf('CRIB        %2.2f\n',mean(-10*log10(crib_A(:))));
fprintf('Exact-conv  %2.2f\n',mean(-10*log10(sae_ttexact(:))));
fprintf('(TT2CPD)ALS %2.2f\n',mean(-10*log10(sae_ttals(:))));
fprintf('CPALS       %2.2f\n',mean(-10*log10(sae_cpals(:))));
