% This file illustrates a CPD of a tensor. 
% Performance of the decomposition is assessed through the squared
% angular error and its Cramer-Rao Induced bound.
%
% Tensorbox v.2015

%% Generate a tensor of size 30 x 30 x 30 and rank 4;
clear all;
N = 3; % tensor order
I = [10 10 10];% tensor size
R = 4; % tensor rank

% Factor matrices
A = cell(N,1);
for n = 1:N
    A{n} = randn(I(n),R);
    A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));
end
lambda = ones(R,1); % lambda = (1:R)';
Y0 = tensor(ktensor(lambda,A));

% Cramer-Rao Induced Bound for estimating A when Gaussian noise has unit
% variance.
% absorb the weights lambda into factor matrices A
A{1} = A{1} * diag(lambda);
cribA = cribCP(A);

%% Cramer-Rao Induced bound on estimating A with unit noise variance 
cribA = cribCP(A,[],1);
    
%% Decompose tensor at different noise levels
SNR = 0:5:30; % Noise level in dB, inf for noiseless tensor
sigma_noise = 10.^(-SNR/20)*(std(double(Y0(:))));

Noruns = 10;
msae_ = zeros(numel(SNR),Noruns);
mse_ = zeros(numel(SNR),Noruns,sum(I)*R);

for ksnr = 1:numel(SNR)
    for krun = 1:Noruns
        fprintf('SNR %d, run %d\n',SNR(ksnr),krun)
        
        %% Add Gaussian noise into the tensor Y
        Noise_ = randn(size(Y0));
        Noise_ = Noise_/std(Noise_(:));
        Y=Y0+sigma_noise(ksnr)*Noise_;
        
        %% Decompose Y using FastALS
        opts = cp_fastals;
        opts.init = 'nvecs';
        
        P = cp_fastals(Y,R,opts);
        
        % Evaluate the Square Angular Error between estimated and orignal
        % loading components
        [msae1,msae2,sae_a,sae2] = SAE(A(:),P.U);
        
        msae_(ksnr,krun) = msae1;
         
    end
end

%% Compare mean squared angular error (MSAE) for all components with mean CRIB

% Compute the mean CRIB at different noise levels
cribA_ = mean(cribA(:)) * sigma_noise(:).^2;
cribA_dB = -10*log10(cribA_);

% Mean SAE over all runs
msae_dB = -10*log10(mean(msae_,2));

figure(1);clf;
h = plot(SNR,[msae_dB cribA_dB]);
set(h,'linewidth',4)
set(h(2),'linestyle','--')
set(h(1),'linestyle','-.')
xlabel('SNR (dB)')
ylabel('MSAE (dB)')
legend('MSAE' , 'CRIB','location','best')

set(gca,'FontSize',18)