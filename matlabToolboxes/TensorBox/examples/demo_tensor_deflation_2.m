% This example illustrates the tensor deflation which sequentially extracts
% rank-1 tensors from a rank-R tensor. 
% 
% The code uses the ASU algorithm.
% 
% TENSOR BOX, v.2015
% 
%% Generate a tensor of size 30 x 30 x 30 and rank 4
clear all

I = 10; %tensor size 
N = 3; % tensor order
R = I; % tensor rank == tensor size

ca = [.0 .7];
A = cell(N,1);
for n = 1:N
    [A{n},D] = gen_matrix(I,R,ca);
    %A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));
end
lambda = ones(R,1);
Y = tensor(ktensor(lambda,A));

% Add Gaussian noise into the tensor Y
SNR = 30; % Noise level in dB, inf for noiseless tensor
sigma_noise = 10^(-SNR/20)*std(double(Y(:)));
noise_ = randn(size(Y));
Y = Y + sigma_noise * noise_/std(noise_(:));
 
% Compute Cramer-Rao Induced Bound for estimating A
cribA = cribCP(A,[],sigma_noise^2); 
 
%% Cramer Rao Bound for tensor deflation
sz = size(Y);
CCRBa_ = zeros(sum(sz),R); CCRIBa_ = zeros(N,R);

% Yoc = tucker_als(Yo,R);
% Ycore = ttm(Yo,Yoc.u,'t');
% A = Ycore.u;
for r = 1:R
    a = cellfun(@(x) x(:,r),A,'uni',0);
    U = cellfun(@(x) x(:,[1:r-1 r+1:end]),A,'uni',0);
    lambda_r = lambda(r);
    G = tendiag(lambda([1:r-1 r+1:end]),(R-1)*ones(1,N));
      
    % Concentrated CRB for deflation with constraints on [w_n, un] and alpha_n
    crl_opt = 3;
    [CCRBa_(:,r),foe,CCRIBa_(:,r)] = crb_bcdLp1(a,U,lambda_r,G,crl_opt);
end

CCRIBa_ = CCRIBa_*sigma_noise^2;

%% ASU
opts = cp_asu;      % get default setting of FastALS
opts.cp_param.init = { 'nvec'  'nvec2'  'osvd' };
% opts.cp_param.refine_a = true;
% opts.cp_param.pre_select = true;
opts.cp_param.alsinit = false;
opts.cp_param.tol = 1e-8;
opts.reductiontype = 'residue';

[P_asu,output] = cp_asu(Y,R,opts); % P is a rank-R K-tensor with N factors 
                        % P.U{1},...,P.U{N} and coefficients P.lambda

P_asu = fixsigns(P_asu);


%% Compute squared angular error and compare with its Cramer-Rao induced bound

[mase_1,mase_,sae] = SAE(A,P_asu.U);
 
fprintf('Mean squared angular error %.3f dB \n',-10*log10(mase_));
fprintf('Cramer-Rao Induced bound (CRIB) for CPD %2.2f dB\n',-10*log10(mean(cribA(:))))
fprintf('CRIB for tensor deflation %2.2f dB\n',-10*log10(mean(CCRIBa_(:))))