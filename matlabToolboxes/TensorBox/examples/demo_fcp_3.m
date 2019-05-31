% This demo will show how the rank-one FCP (R1FCP) is sentive to unfolding
% rules. With a ``bad'' unfolding, R1FCP may completely fail to estimate
% latent components from the unfolded tensor.
% 
% The example also shows that (low-rank) FCP is much better than R1FCP.
% 
% REF:
%
% [1] Anh-Huy  Phan, Petr Tichavsky, Andrzej Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", available
% on line at http://arxiv.org/abs/1211.3796, 2012  
%
%
% See also: cp_fcp, cp_fastals
%
% The demo is a part of the TENSORBOX, 2012.
%
% Phan Anh Huy, March 2013
% phan@brain.riken.jp
% 
% The authors would be grateful for feedback and suggestions on this demo
% and the FCP algorithm.

clear all; warning off;
N = 5;      % tensor order
R = 15;  I = R * ones(1,N); %tensor rank and tensor size

% Parameters of the FastALS algorithm
tol = 1e-8;
maxiters = 1000;

% Generate factors An with collinearity coefficients in specific ranges
cn =  [0.  .45   
    .2 .65
    .5 .99
    .95 .99
    .9 .99];    

A = cell(N,1);
for n = 1:N
    A{n} = gen_matrix(I(n),R,cn(n,:)); % % A(:,r)' * A(:,s) in [c_n(1),c_n(2)]
end
Y = ktensor(A(:));      % tensor in the Kruskal form
    
% Add noise to tensor Y 
SNR = 0;               % Noise level (dB)
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(I));
Y = full(Y);
if ~isinf(SNR)
    Y = Y + sig2 * randn(I);
end
normY = norm(Y); % Frobenious norm of Y


% plot collinear coefficients of An
Ca = zeros(R,R,N);
for n = 1:N
    Ca(:,:,n) = A{n}'*A{n};
end
Crange = reshape(abs(Ca),[],N);
Crange(1:R+1:end,:) = [];

fig = figure(1); set(gca,'fontsize',16); clf
corr_bin = cos((90:-1:0)/180*pi);
aux = hist(Crange,corr_bin);
h = plot(corr_bin,bsxfun(@rdivide,aux,sum(aux)));

xlabel('c_n')
ylabel('Density')
legend(h,arrayfun(@(x) sprintf('C%d',x),1:N,'uni',0),'location','best')


%% Cramer-Rao induced bound on factor matrices
try
    if ~isinf(SNR)
        crib = cribCP(A,[],sig2^2);
        fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))
    end
catch
    warning('CRIB is not available.')
end

%% Generate all possible unfoldings for order-5 tensor
% There are two types of unfoldings to reshape an order-5 tensor to be
% order-3.
%  - [1,1,3]-type: combines 3 modes
%  - [1,2,2]-type: combines 2 modes

clear foldrule;
% [1,1,3]-type
frule = nchoosek(1:N,2);
for kf = 1:size(frule,1)
    foldrule{kf} = [mat2cell(frule(kf,:),1,ones(1,2)) setdiff(1:N,frule(kf,:))];
end

% [1,2,2]-type
clear foldrule2
frule = nchoosek(1:N,1);
cnt = 1;
for kf = 1:size(frule,1)
    pat1 = setdiff(1:N,frule(kf,:));
    frule2 = nchoosek(pat1(2:end),1);
    for kf2 = 1:size(frule2,1)
        foldrule2{cnt} = [{frule(kf,:)} [pat1(1) frule2(kf2,:)] setdiff(pat1(2:end),frule2(kf2,:))];
        cnt = cnt+1;
    end
end

% There are in total 25 unfolding rules
UnfoldingRule = [foldrule  foldrule2];

%% Get and set parameters of FCP
opts = cp_fcp;
opts.compress_param.compress = true;
opts.var_thresh = 0; % 0<= var_thresh <=1, rank-1 to lowrank FCP

opts.cp_func = @cp_fastals; % CP algorithm to decompose unfolded tensor
opts.cp_param = feval(opts.cp_func); % get parameters of FastALS
opts.cp_param.init = 'dtld'; % Multiinitialization
opts.cp_param.linesearch = false;
opts.cp_param.printitn = 0;
opts.cp_param.tol = tol;
opts.cp_param.maxiters = maxiters;

%% Run Rank-1 FCP over all unfolding rules. This process may be time consuming.
for rule = 1:numel(UnfoldingRule)
    
    % Get and set parameters of FCP 
    opts.foldingrule = UnfoldingRule{rule}; 
    opts.var_thresh = 0; % 0<= var_thresh <=1, rank-1 to lowrank FCP
    
    ts = tic;
    [P,output{rule},BagofOut{rule}] = cp_fcp(Y,R,opts);
    t(rule,1) = toc(ts);

    % Evaluate performance
    P = arrange(P);
    msae(rule,1) = SAE(A,P.U);
    fit(rule,1) = 1- sqrt(normY^2 + norm(P)^2 - 2 * innerprod(Y,P))/normY;
    fprintf('Rule %s MSAE %.2f, Fit %2.2d, Exec.time %.3f (s)\n',...
        foldingrule2char(opts.foldingrule),...
        -10*log10(msae(rule,1)),fit(rule,1), t(rule,1));
    
end

%% Run (low-rank) FCP over all unfolding rules
for rule = 1:numel(UnfoldingRule)
    
    %% Get and set parameters of FCP 
    opts.foldingrule = UnfoldingRule{rule}; 
    opts.var_thresh = 0.9999; % low-rank FCP
    
    ts = tic;
    [P,output{rule},BagofOut{rule}] = cp_fcp(Y,R,opts);
    t(rule,2) = toc(ts);
    
    % Evaluate performance
    P = arrange(P);
    msae(rule,2) = SAE(A,P.U);
    fit(rule,2) = 1- sqrt(normY^2 + norm(P)^2 - 2 * innerprod(Y,P))/normY;
    fprintf('Rule %s MSAE %.2f, Fit %2.2d, Exec.time %.3f (s)\n',...
        foldingrule2char(opts.foldingrule),...
        -10*log10(msae(rule,2)),fit(rule,2), t(rule,2));   
end

%% Visualize MSAEs of rank-one FCP and (low-rank) FCP for different unfolding rules

fig = figure(2); clf; set(gca,'fontsize',16)

% Unfolding types
foldstr = cellfun(@foldingrule2char,UnfoldingRule,'uni',0);
ftype = cell2mat(cellfun(@(x) cellfun(@numel,x)' ,UnfoldingRule,'uni',0))';
ftypeset = unique(ftype,'rows');

% Sort msae, fit in ascesding order in each group of unfolding
msaedb = -10*log10(msae);
reordii = [];
for kset = 1:size(ftypeset,1)
    ifkset = find(all(bsxfun(@eq,ftype,ftypeset(kset,:)),2));
    
    [val,ii] = sortrows(msaedb(ifkset),1);
    reordii = [reordii;ifkset(ii)];
end
fit = fit(reordii,:);msaedb = msaedb(reordii,:);t = t(reordii,:);
foldstr = foldstr(reordii);

% Plot CRIB line
mcrib = mean(-10*log10(crib(:)));
hcr = plot(1:numel(UnfoldingRule),mcrib*ones(1,numel(UnfoldingRule)),...
    'k--','linewidth',2); hold on

% Plot MSAE of R1-FCP and FCP
h = plot(1:size(msae,1),msaedb);
set(h,'linewidth',2)
set(gca,'fontsize',16)

% Good and bad unfoldings for R1-FCP
[mval,ind] = sort(msaedb(:,1));

hw = plot(ind(1:3),mval(1:3),'or','markersize',14,'linewidth',2);
hb = plot(ind(end-5:end),mval(end-5:end),'sm','markersize',14,'linewidth',2);

set(gca,'position',[ 0.1119    0.3214    0.8131    0.6036])
% xlabel('Unfolding rule')
ylabel('MSAE (dB)')
hleg = legend([hcr; h(:);hw; hb],{'CRIB' 'R1FCP' 'FCP' 'R1FCP, bad folding' ...
    'R1FCP, good folding'},'location','best');
set(hleg,'color','none')

axis auto

set(gca,'xtick',1:numel(UnfoldingRule))
hText = RotateXLabel(90,foldstr);
set(hText,'fontsize',14)
 
% fn = sprintf('fig_msae_N%dI%dR%d_SNR%d_crange_fit',N,I(1),R,SNRa);
% saveas(fig,fn,'fig')
% fn = [fn '.eps'];
% saveas(fig,fn,'epsc')
% fixPSlinestyle(fn,fn)

% "Bad" unfoldings which combine two modes with lowest collinearity degrees,
% i.e., mode-1 and mode-2, cause significant loss of accuracy for R1FCP.