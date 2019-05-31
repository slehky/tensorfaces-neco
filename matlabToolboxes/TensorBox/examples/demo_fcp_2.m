% This file illustrates the FCP algorithm for decomposition of a high order
% tensor with "good" and "bad" tensor unfoldings.
%
% % REF:
%
% [1] Anh-Huy  Phan, Petr Tichavsky, Andrzej Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012  
%
% [2] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, "Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition", IEEE TSP, in print,
% 2013, available online at http://arxiv.org/abs/1209.3215, 2012. 
%
% See also: cp_fcp, cp_fLM, cp_fastals, cp_als
%
% The demo is a part of the TENSORBOX, 2012.
%
% Phan Anh Huy, April 2012
% phan@brain.riken.jp
% 
% The authors would be grateful for feedback and suggestions on this demo
% and the FCP algorithm.

%% The following results are for decomposition of order-5 tensor In = R = 10
% SNR = 20dB.
%
% Algorithm             MSAE (dB)  Fit      R.Time (seconds)
% ALS                    46.38,    0.9006,  1.87  
% R1FCP [(1,2,3), 4, 5]  32.71,    0.8541,  1.42 	(bad tensor unfolding)
% R1FCP [1, 2, (3,4,5)]  46.03,    0.9006,  0.38 	(good tensor unfolding)
% FCP   [(1,2,3), 4, 5]  43.21,    0.9005,  1.54 	(bad tensor unfolding)
% FCP   [1, 2, (3,4,5)]  46.03,    0.9006,  0.53 	(good tensor unfolding)


%% Generate tensor 
clear all

% Generate (I x R) factor matrices An with specific correlation coefficients
c = [.1 .1 .9 .99 .99 ];    % correlation coefficients
N = numel(c);            % tensor order 
R = 10;                  % Number of components
I = R* ones(1,N);        % tensor size

A = cell(N,1);Ca = zeros(R,R,N);
for n = 1:N
    A{n} = gen_matrix(I(n),R,c(n)); % % A(:,r)' * A(:,s) = c_n
    Ca(:,:,n) = A{n}'*A{n}; 
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

tol = 1e-10;
maxiters = 1000;

%% Cramer-Rao induced bound on factor matrices
try
    if ~isinf(SNR)
        crib = cribCP(A,[],sig2^2);
        fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))
    end
catch
    warning('CRIB is not available.')
end

%% ALS
ts = tic;
Yals = cp_als(Y,R,'init','nvecs','tol',tol,'maxiters',maxiters);
tals = toc(ts);


% Squared angular error and fit
msaeals = SAE(A,Yals.U);
fitals = 1- sqrt(normY^2 + norm(Yals)^2 - 2 * innerprod(Y,Yals))/normY;

% fprintf('ALS, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
%     -10*log10(msaeals),fitals, tals);

%% FastALS
% opts = cp_fastals;
% opts.maxiters = maxiters;
% opts.printitn = 0;
% opts.init = 'nvecs';
% opts.tol = tol;
% ts = tic;
% Yfals = cp_fastals(Y,R,opts); % fast ALS using fast CP gradient
% tfals = toc(ts);
% 
% % Squared angular error and fit
% msaefals = SAE(A,Yfals.U);
% fitfals = 1- sqrt(normY^2 + norm(Yfals)^2 - 2 * innerprod(Y,Yfals))/normY;
% 
% % fprintf('ALS, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
% %     -10*log10(msaefals),fitfals, tfals);

%% Rank-one FCP with "bad" unfolding rule using FastALS for order-3 tensor
% [foldingrule,cfold] = unfoldingstrategy(A,3);

% parameters for FCP
opts = cp_fcp; % default values

% parameters for unfolding and low rank approximation
badfolding = {[1 2 3] 4 5}; % a "bad" unfolding combines modes which have  
                            % lowest collinearity degrees
opts.foldingrule = badfolding; % [1 1 2]
opts.var_thresh = .0; % set threshold to 0.99 for low rank approximation

% parameters for Tucker compression
opts.compress_param.compress = true;

% parameters for CP algorithm
opts.cp_func = @cp_fastals; % @cp_fLMa_v2; % change here for other CP algorithms
opts.cp_param = feval(opts.cp_func);
opts.cp_param.init = 'dtld';    
opts.cp_param.tol = tol;
opts.cp_param.maxiters = maxiters;

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
t = toc(ts);
Yhat = arrange(Yhat);

% Squared angular error and fit
msae = SAE(A,Yhat.U);
fit = 1- sqrt(normY^2 + norm(Yhat)^2 - 2 * innerprod(Y,Yhat))/normY;


% fprintf('Rule %s, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
%     foldingrule2char(opts.foldingrule),-10*log10(msae),fit, t);

%% Rank-one FCP with a "good" unfolding rule

goodfolding = {1 2 [3 4 5]};
opts.foldingrule = goodfolding; % [1 1 2]

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
tgood = toc(ts);
Yhat = arrange(Yhat);
% Squared angular error
msaegood  = SAE(A,Yhat.U);
fitgood = 1- sqrt(normY^2 + norm(Yhat)^2 - 2 * innerprod(Y,Yhat))/normY;

% fprintf('Rule %s, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
%     foldingrule2char(opts.foldingrule),-10*log10(msaegood),fitgood, tgood);

%% FCP with a "bad" unfolding
opts.foldingrule = badfolding; % [1 1 2]
opts.var_thresh = .9999; % set threshold to 0.99 for low rank approximation

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
tfcpb = toc(ts);
Yhat = arrange(Yhat);

% Squared angular error and fit
msaefcpb = SAE(A,Yhat.U);
fitfcpb = 1- sqrt(normY^2 + norm(Yhat)^2 - 2 * innerprod(Y,Yhat))/normY;

%% FCP with a "good" unfolding rule
goodfolding = {1 2 [3 4 5]};
opts.foldingrule = goodfolding; % [1 1 2]
opts.var_thresh = .999; % set threshold to 0.99 for low rank approximation

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
tfcpgood = toc(ts);
Yhat = arrange(Yhat);
% Squared angular error
msaefcpgood  = SAE(A,Yhat.U);
fitfcpgood = 1- sqrt(normY^2 + norm(Yhat)^2 - 2 * innerprod(Y,Yhat))/normY;


%% Compare ALS and FCP
clc
ch_badfolding = foldingrule2char(badfolding);
ch_goodfolding = foldingrule2char(goodfolding);
mch = max(numel(ch_badfolding),numel(ch_goodfolding));

cprintf('_blue','Algorithm %s MSAE (dB)  Fit      R.Time (seconds)\n',...
    repmat(' ',1,mch-4));

fprintf('ALS  %s %.2f,    %.4f,  %.2f  \n',...
    repmat(' ',1,mch+2),-10*log10(msaeals),fitals, tals);
% fprintf('FastALS            %.2f dB,  %.4f,  %.2f  \n',...
%     -10*log10(msaefals),fitfals, tfals);
fprintf('R1FCP %s  %.2f,    %.4f,  %.2f \t(bad tensor unfolding)\n',...
    foldingrule2char(badfolding),-10*log10(msae),fit, t);
cprintf('red','R1FCP %s  %.2f,    %.4f,  %.2f \t(good tensor unfolding)\n',...
    foldingrule2char(goodfolding),-10*log10(msaegood),fitgood, tgood);

fprintf('FCP   %s  %.2f,    %.4f,  %.2f \t(bad tensor unfolding)\n',...
    foldingrule2char(badfolding),-10*log10(msaefcpb),fitfcpb, tfcpb);
cprintf('red','FCP   %s  %.2f,    %.4f,  %.2f \t(good tensor unfolding)\n',...
    foldingrule2char(goodfolding),-10*log10(msaefcpgood),fitfcpgood, tfcpgood);
cprintf('black','')

fprintf('Cramer-Rao Induced bound %2.2f dB\n',mean(-10*log10(mean(crib,2))))


%% Visualize FCP output as function of execution time
vis_fcp_output(BagofOut)