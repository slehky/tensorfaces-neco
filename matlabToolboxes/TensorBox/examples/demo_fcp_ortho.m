% CPD with one column-wise orthogonal factor matrix using the FCP
% algorithm.
%
% The example compares the CPO-ALS2 algorithm and the FCP algorithm
%  - In CPO-ALS2 
%      + orthogonal factor matrices are estimated from SVD of CPgradients
%      + other factors are estimated from rank-one tensor approximations.
%
%  - In FCP
%      + UnFold the tensor to be lower order such that orthogonal modes
%      are still preserved.
%      + Decompose the unfolded tensor using CPO-ALS.
%      + Construct and decompose structured Kruskal tensor using a variant
%      of CPO-ALS for structured K-tensor.
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
% See also: cp_fcp, cpo_als1, cpo_als2, demo_CPD_5
%
% The demo is a part of the TENSORBOX, 2012.
%
% Phan Anh Huy, April 2012
% phan@brain.riken.jp
% 
% The authors would be grateful for feedback and suggestions on this demo
% and the FCP algorithm.

%% Generate tensor 
clear all; warning off
N = 5;                   % tensor order 
R = 15;                  % Number of components
I = R* ones(1,N);        % tensor size

tol = 1e-8;
maxiters = 1000;

% Generate (I x R) matrices A with collinearity coefficients distributed in
% specific ranges
cn = [0 0       % A1 comprises orthogonal components, whereas
    .9 .99      % components of A2,...,A5 are highly collinear
    .9 .99     
    .9 .99
    .9 .99];    

A = cell(N,1);
for n = 1:N
    if n == 2
        A{n} = gen_matrix(I(n),R,cn(n,:),2); % A(:,r)'*A(:,s) in [c_n(1),c_n(2)]
    else
        A{n} = gen_matrix(I(n),R,cn(n,:)); % A(:,r)'*A(:,s) in [c_n(1),c_n(2)]
    end
end
Y = ktensor(A(:));      % tensor in the Kruskal form
    
% Add noise to tensor Y 
SNR = 0;               % Noise level (dB)
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(I));
Y = full(Y);
if ~isinf(SNR)
    Y = Y + sig2 * randn(I);
end
normY = norm(Y);


% Plot collinear coefficients of An
Ca = zeros(R,R,N);
for n = 1:N
    Ca(:,:,n) = A{n}'*A{n};
end
Crange = reshape(abs(Ca),[],N);
Crange(1:R+1:end,:) = [];
fig = figure(1); clf;set(gca,'fontsize',16)
corr_bin = cos((90:-1:0)/180*pi);
aux = hist(Crange,corr_bin);
h = plot(corr_bin,bsxfun(@rdivide,aux,sum(aux)));

xlabel('c_n')
ylabel('Density')
legend(h,arrayfun(@(x) sprintf('C%d',x),1:N,'uni',0),'location','best')


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

%% CPO-ALS2
cp_param = cpo_als2;
cp_param.tol = tol;
cp_param.maxiters = maxiters;
cp_param.init = 'nvecs';
cp_param.printitn = 1;
cp_param.orthomodes = 1; % set the orthogonal factor matrix

ts = tic;
[Yd,outputFALS] = cpo_als2(Y,R,cp_param);
tals = toc(ts);

msaeals = SAE(A,Yd.U);
fitals = 1- sqrt(normY^2 + norm(Yd)^2 - 2 * innerprod(Y,Yd))/normY;

fprintf('CPO-ALS2, MSAE %.2f dB,  Fit %.2f, R.Time %.2f seconds \n',...
    -10*log10(msaeals),fitals, tals);


%% Compare ALS and FCP
clc
ch_goodfolding = foldingrule2char(opts.foldingrule);
mch = numel(ch_goodfolding);

cprintf('_blue','Algorithm %s MSAE       Fit      R.Time (seconds)\n',...
    repmat(' ',1,mch-3));

fprintf('CPO-ALS2  %s %.2f dB,  %.4f,  %.2f  \n',...
    repmat(' ',1,mch-3),-10*log10(msaeals),fitals, tals);
fprintf('FCP   %s  %.2f dB,  %.4f,  %.2f\n',...
    ch_goodfolding,-10*log10(msaefcp),fitfcp, tfcp);


%% Visualize FCP output
vis_fcp_output(BagofOut)