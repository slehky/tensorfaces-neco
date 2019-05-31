% CPD of complex-valued tensor with one column-wise orthogonal factor matrix. 
%
% The example compares CPO-ALS1, CPO-ALS, and FCP using CPO-ALS2 and
% CPO-ALS for structured CPD. 
% 
% REF:
%
% [1] Anh-Huy  Phan, Petr Tichavsky, Andrzej Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012  
% 
% [2] M. Sorensen, L. Lathauwer, P. Comon, S. Icart, and L. Deneire,
% "Canonical polyadic decomposition with a columnwise orthonormal
% factor matrix,"SIAM Journal on Matrix Analysis and Applications,
% vol. 33, no. 4, pp. 1190-1213, 2012.
%
% See also: cp_fcp, cpo_als1, cpo_als2, demo_fcp_ortho
%
% The demo is a part of the TENSORBOX, 2012.
%
% Phan Anh Huy, April 2012
% phan@brain.riken.jp
% 
% The authors would be grateful for feedback and suggestions on this demo
% and the FCP algorithm.

%% Generate tensor 
clear all
N = 5;                   % tensor order
R = 10;                  % Number of components
I = R* ones(1,N);        % tensor size

tol = 1e-8;
maxiters = 1000;

% Generate a Kruskal tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
c = [0 0 ;
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
normY = norm(Y);


%% FCP with orthogonality constrains on the first factor matrix
opts = cp_fcp;
opts.compress_param.compress = true;

opts.var_thresh = 0.9999;
opts.TraceFit = true;
opts.foldingrule = {1 [2 3] [4 5]};
opts.cp_func = @cpo_als2;              % Set algorithm for unfolded tensor
opts.cp_param = feval(opts.cp_func);
opts.cp_param.init = 'dtld';
opts.cp_param.printitn = 0;
opts.cp_param.tol = tol;
opts.cp_param.maxiters = maxiters;
opts.cp_param.orthomodes = 1;
opts.cpstruct_func = @cpostruct_als;    % Set algorithm for structured CPD

ts = tic;
[P3,outputr1,BagofOut] = cp_fcp(Y,R,opts);
tfcp = toc(ts);

msaefcp  = SAE(A,P3.U);
fitfcp = 1- sqrt(normY^2 + norm(P3)^2 - 2 * innerprod(Y,P3))/normY;

fprintf('Rule %s, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
    foldingrule2char(opts.foldingrule),-10*log10(msaefcp),fitfcp, tfcp);

%% CPO-ALS1
cp_param = cpo_als1;
cp_param.tol = tol;
cp_param.maxiters = maxiters;
cp_param.init = 'nvecs';
cp_param.printitn = 1;
cp_param.orthomodes = 1; % set the orthogonal factor matrix

ts = tic;
[Yd,outputALS] = cpo_als1(Y,R,cp_param);
tals1 = toc(ts);

msaeALS1 = SAE(A,Yd.U);
fitals1 = outputALS.Fit(end);

fprintf('CPO-ALS2, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
    -10*log10(msaeALS1),fitals1, tals1);

%% CPO-ALS2
cp_param = cpo_als2;
cp_param.tol = tol;
cp_param.maxiters = maxiters;
cp_param.init = 'nvecs';
cp_param.printitn = 1;
cp_param.orthomodes = 1; % set the orthogonal factor matrix

ts = tic;
[Yd,outputALS] = cpo_als2(Y,R,cp_param);
tals2 = toc(ts);

msaeALS2 = SAE(A,Yd.U);
fitals2 = outputALS.Fit(end);

fprintf('CPO-ALS2, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
    -10*log10(msaeALS2),fitals2, tals2);


%% Compare ALS and FCP
clc
ch_goodfolding = foldingrule2char(opts.foldingrule);
mch = numel(ch_goodfolding);

cprintf('_blue','Algorithm %s MSAE       Fit      R.Time (seconds)\n',...
    repmat(' ',1,mch-3));

fprintf('CPO-ALS1  %s %.2f dB,  %.4f,  %.2f  \n',...
    repmat(' ',1,mch-3),-10*log10(msaeALS1),fitals1, tals1);
fprintf('CPO-ALS2  %s %.2f dB,  %.4f,  %.2f  \n',...
    repmat(' ',1,mch-3),-10*log10(msaeALS2),fitals2, tals2);
fprintf('FCP   %s  %.2f dB,  %.4f,  %.2f\n',...
    ch_goodfolding,-10*log10(msaefcp),fitfcp, tfcp);


%% Visualize FCP output
vis_fcp_output(BagofOut)