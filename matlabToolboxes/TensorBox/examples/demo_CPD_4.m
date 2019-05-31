% This demo shows CP decomposition of a complex-valued tensor using the
% FastALS, fLM and FCP algorithms.
%
% REF:
%
% [1] Anh-Huy Phan, P. Tichavsky, and Andrzej Cichocki, "On fast computation 
% of gradients for CP algorithms", http://arxiv.org/abs/1204.1586, 2012, 
%
% [2] A.-H. Phan, P. Tichavsky, A. Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", available
% online at http://arxiv.org/abs/1211.3796, 2012
%
% [3] A.-H. Phan, P. Tichavsky, A. Cichocki, "Low Complexity Damped
% Gauss-Newton Algorithms for CANDECOMP/PARAFAC", SIAM, Journal on Matrix 
% Analysis and Applications, vol. 34, pp. 126?147, 2013.
%
% See also: cp_fcp, cp_fastals, cp_fLM
%
% TENSORBOX, 2013.
% Phan Anh Huy, March 2013
% phan@brain.riken.jp
% 
%% Generate order-5 tensor 
clear all
N = 5;                   % tensor order
R = 10;                  % Number of components
I = R* ones(1,N);        % tensor size

% Generate a Kruskal tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
c = [.02 .3;
     .3 .6
     .4 .8
     .9 .99
     .9 .99]; % collinearity degrees of factor matrices
A = cell(N,1);
for n = 1:N
    [foe,D] = gen_matrix(R,R,c(n,:)); % % A(:,r)' * A(:,s) = c_n
    A{n} = randn(I(n),R) + 1i*randn(I(n),R);
    A{n} = orth(A{n}) * D;
end

% Plot density of collinearity coefficients of An, i.e. abs(An(:,r)*An(:,s))
Ca = zeros(R,R,N);
for n = 1:N
    Ca(:,:,n) = A{n}'*A{n};
end
fig = figure(1); clf; set(gca,'fontsize',16)
Crange = reshape(abs(Ca),[],N);
Crange(1:R+1:end,:) = [];
corr_bin = cos((90:-1:0)/180*pi);
aux = hist(Crange,corr_bin);
h = plot(corr_bin,bsxfun(@rdivide,aux,sum(aux)));

xlabel('c')
ylabel('Density')
legend(h,arrayfun(@(x) sprintf('C%d',x),1:N,'uni',0),'location','best')


Y = ktensor(A(:));      % tensor in the Kruskal form
    
% Add noise to tensor Y 
SNR = 20;               % Noise level (dB)
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(I));
Y = full(Y);
if ~isinf(SNR)
    Y = Y + sig2 * (randn(I) + 1i* randn(I))/sqrt(2);
end

%% FastALS
opts = cp_fastals;    % get parameters of FastALS
opts.init = 'dtld';   % use extended gram or direct trilinear decomposition
opts.linesearch = 0;
opts.printitn = 1;
opts.tol = 1e-8;  
opts.maxiters = 1000;

ts= tic;
[P,output] = cp_fastals(Y,R,opts);
tals = toc(ts);

% Evaluate squared angular errors
msaeals = SAE(A,P.U);
fitals = real(output.Fit(end,2));
 
cprintf('red','FastALS  MSAE %.2f dB, Fit %.4d, Execution time %.2f (s)\n',...
    -10*log10(msaeals),fitals, tals);
cprintf('black','')


%% fLM
opts = cpx_fLMa;    % get parameters of FastALS
opts.init = 'dtld';   % use extended gram or direct trilinear decomposition
opts.printitn = 1;
opts.tol = 1e-8;  
opts.maxiters = 1000;
opts.MaxRecursivelevel = 0;
opts.maxboost = 0;
opts.updaterule = 2;

ts= tic;
[P,output] = cpx_fLMa(Y,R,opts);
tflm = toc(ts);

% Evaluate squared angular errors
msaeflm = SAE(A,P.U);
fitflm = real(output.Fit(end,2));
 
cprintf('red','FastALS  MSAE %.2f dB, Fit %.4d, Execution time %.2f (s)\n',...
    -10*log10(msaeflm),fitflm, tflm);
cprintf('black','')

%% FCP with unfolding rule [1, 2, (3,4,5)] 
opts = cp_fcp;      % get default parameters of FCP

opts.foldingrule = {1 2 [3 4 5]}; % reshape an order-5 to order-3 tensor
opts.var_thresh = .9999;  

opts.cp_func = @cp_fastals; % @cp_fLMa; % change here for other CP algorithms
opts.cp_param = feval(opts.cp_func);    % get parameters of FastALS
opts.cp_param.init = 'dtld';            % See document of FastALS
opts.cp_param.tol = 1e-8;  
opts.cp_param.maxiters = 1000;

% Execute algorithm
ts = tic;
[Yhat,output,BagofOut] = cp_fcp(Y,R,opts);
tfcp = toc(ts);
Yhat = arrange(Yhat);

% Evaluate performance through squared angular error
msaefcp  = SAE(A,Yhat.U);
fitfcp = real(BagofOut{end}.Fit(end,end));
 
cprintf('red','FCP %s  MSAE %.2f dB, Fit %.4f, Execution time %.2f (s)\n',...
    foldingrule2char(opts.foldingrule),-10*log10(msaefcp),fitfcp, tfcp);
cprintf('black','')

%% Summarize results
clc
folding = foldingrule2char({1 2 [3 4 5]});
folding(folding==' ') = [];
mch = numel(folding);

cprintf('_blue','Algorithm %s MSAE (dB)     Fit      R.Time (seconds)\n',...
    repmat(' ',1,mch-5));

fprintf('ALS  %s %.2f,       %.4f,  %.2f  \n',...
    repmat(' ',1,mch)+2,-10*log10(msaeals),fitals, tals);

fprintf('fLM  %s %.2f,       %.4f,  %.2f  \n',...
    repmat(' ',1,mch)+2,-10*log10(msaeflm),fitflm, tflm);

fprintf('FCP %s  %.2f,       %.4f,  %.2f \n',...
    folding,-10*log10(msaefcp),fitfcp, tfcp);

 