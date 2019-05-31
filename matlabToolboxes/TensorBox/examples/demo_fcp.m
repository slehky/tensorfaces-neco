% This file shows how to use the FCP algorithm for decomposition of a high
% order tensor through tensor unfolding.
%
% REF:
%
% [1] Anh-Huy  Phan, Petr Tichavsky, Andrzej Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012  
%
% [2] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, "Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition", IEEE TSP, in print,
% 2013, available online at http://arxiv.org/abs/1209.3215, 2012. 
%
% [3] Anh-Huy Phan, P. Tichavsky, and Andrzej Cichocki, "On fast computation of
% gradients for CP algorithms", http://arxiv.org/abs/1204.1586, 2012, 
%
%
% See also: cp_fcp, cp_fastals, cp_fLM, demo_fcp_2, demo_fcp_3
%
% TENSORBOX, 2012.
% Phan Anh Huy, April 2012
% phan@brain.riken.jp
% 
%% Generate tensor 
clear all; warning off;

% Generate (I x R) factor matrices A with specific correlation coefficients
c = [.1 .1 .9 .99 .99 ]; % collinearity degrees of factor matrices
N = numel(c);            % tensor order 
R = 10;                  % Number of components
I = R* ones(1,N);        % tensor size

A = cell(N,1);           
for n = 1:N
    A{n} = gen_matrix(I(n),R,c(n)); % % A(:,r)' * A(:,s) = c_n
end
Y = ktensor(A(:));      % tensor in the Kruskal form
    
% Add noise to tensor Y 
SNR = 20;               % Noise level (dB)
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(I));
Y = full(Y);
if ~isinf(SNR)
    Y = Y + sig2 * randn(I);
end

normY = norm(Y);

%% Cramer-Rao induced bound on factor matrices
try
    if ~isinf(SNR)
        crib = cribCP(A,[],sig2^2);
        fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))
    end
catch
end


%% FCP with a "good" unfolding rule
% parameters for FCP
opts = cp_fcp; % default values

% parameters for tensor unfolding and low rank approximation
% Unfolding rules [1,2,(3,4,5)] combines modes 3,4 and 5 to reshape an
% order-5 tensor to be order-3. 
% Note: Other unfoldings, e.g., [(1 5), (2 4) 3], [2,3,(1,4,5)] are also
% suitable. However, do not unfold modes 1 and 2 which have low
% collinearity degrees. 

opts.foldingrule = {1,2,[3,4,5]}; %{1 2 [3 4 5]};
opts.var_thresh = .999; % set threshold to 0 for rank-1 approximation

% parameters for Tucker compression
opts.compress_param.compress = true; % set true for a prior Tucker compression 
                                     % if rank-R is lower than tensor size In

% parameters for CP algorithm
opts.cp_func = @cp_fastals; % @cp_fLMa; % change here for other CP algorithms
opts.cp_param = feval(opts.cp_func);    % get parameters of FastALS
opts.cp_param.init = 'nvecs';           % See document of FastALS
opts.cp_param.tol = 1e-10;
opts.cp_param.maxiters = 1000;

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
tgood = toc(ts);
Yhat = arrange(Yhat);

% Evaluate performance through Squared angular error
msaegood  = SAE(A,Yhat.U);
fitgood = 1- sqrt(normY^2 + norm(Yhat)^2 - 2 * innerprod(Y,Yhat))/normY;
 
cprintf('red','FCP %s  MSAE %.2f dB, Fit %.4f, Execution time %.2f (s) (good tensor unfolding)\n',...
    foldingrule2char(opts.foldingrule),-10*log10(msaegood),fitgood, tgood);
cprintf('black','')
fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))


%% Visualize FCP output
vis_fcp_output(BagofOut)