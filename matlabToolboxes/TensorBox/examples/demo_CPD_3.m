% This demo compares intialization methods using in CP algorithms.
% 1- 'dtld' :  Direct Trilinear decomposition and its extended version for
%              higher order CPD using the FCP algorithm
%              XDTLD is highly suggested for most algorithms for CPD.
%
% 2- 'nvec' :  SVD-based initialization using leading singular vectors of
%              mode-n unfoldings 
% 3- 'random': random initialization
% 4- 'ortho':  orthogonal factor matrices randomly generated (In>=R)
% 5- 'fiber'   select fibers from the data tensor.
% 6- Multi-initialization
%
% Initializations for CPD are implemented in the Matlab routine cp_init.
%
% See: cp_init, cp_gram, cp_fastals, 
%
% TENSORBOX, 2013
% Phan Anh Huy,  phan@brain.riken.jp
%
clear all
warning off
N = 6;   % tensor order
R = 10;   % tensor rank
I = ones(1,N) * R; % tensor size


% Generate a Kruskal tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
% c = [0 0.3; 0.1 0.4; 0.5 0.99 ; 0.9 0.99; 0.9 0.999;0.9 0.99; 0.9 0.999];
c = [0.9 0.999;  0.9 0.999;  0.9 0.999;  0.9 0.999; 0.9 0.999;0.9 0.999];

A = cell(N,1);
for n = 1:N
    A{n} = gen_matrix(I(n),R,c(n,:));
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

% Add Gaussian noise
Y = ktensor(A(:));  % a Kruskal tensor of rank-R
SNR = 20; % Noise level in dB, set SNR = inf for noiseless tensor
normY = norm(Y);
if ~isinf(SNR)
    Y = full(Y);
    sig2 = normY./10.^(SNR/20)/sqrt(prod(I));
    Y = Y + sig2 * randn(I);
end
Y = tensor(Y);

%% Cramer-Rao induced bound on factor matrices
try
    if ~isinf(SNR)
        crib = cribCP(A,[],sig2^2);
        fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))
    end
catch;
end

%% Parameters for FastALS
opts = cp_fastals;
opts.linesearch = false;
opts.printitn = 1;
opts.tol = 1e-8;
opts.maxiters= 2000;

exectime = [];
Fit = [];
MSAE = [];
hline = [];
algs = {};


%% eXtended DTLD for CPD

for k = 1:6
    switch k
        case 1
            opts.init = 'dtld';
        case 2
            opts.init = 'fiber';
        case 3
            opts.init = 'random';
        case 4
            opts.init = 'nvecs';
        case 5
            opts.init = 'orth';            
        case 6
            opts.init = {'fiber' 'fiber' 'random' 'random'}; % 'dtld' can also be included 
    end
    %profile -memory;profile on
    tic;
    [P,output] = cp_fastals(Y,R,opts);
    t = toc;
    %profile off
    
    msae = SAE(A,P.U);
    fit = real(output.Fit(end,2));
    
    fprintf('Mean squared angular error %.3f dB \nFit %.4f \nExecution time %.3f seconds\n',...
        -10*log10(msae),fit,t);
    
    exectime(k) = t;
    algs{k} = opts.init;
    Fit{k} = output.Fit;
    MSAE(k) = msae;
end


%% Visualize and Compare results

figure(1);clf; set(gca,'fontsize',16);hold on
clrorder = get(gca,'colorOrder');
for k = 1: numel(Fit)
    h(k) = plot(Fit{k}(:,1),1-real(Fit{k}(:,2)),'color',clrorder(k,:));
end
set(h,'linewidth',2)
set(gca,'yscale','log')
set(gca,'xscale','log')

xlabel('No. Iterations')
ylabel('Relative Error')    
axis tight
algs2 = algs;
algs2(cellfun(@iscell,algs)) = {'multi'};
legend(h,algs2);


[foe,fstalg] = min(exectime);

cprintf('_blue','Algorithm    Exec.time (sec)  Rel. Error   MSAE (dB)\n');
for k = 1: numel(exectime)
    if k == fstalg
        cprintf('red*','%-10s   %5s           %e   %2.2f\n',...
            algs2{k},sprintf('%-2.2f',exectime(k)),1-real(Fit{k}(end,2)),-10*log10(MSAE(k)));
    else
        fprintf('%-10s   %5s           %e   %2.2f\n',...
            algs2{k},sprintf('%-2.2f',exectime(k)), 1-real(Fit{k}(end,2)),-10*log10(MSAE(k)))
    end
end
cprintf('black','')
fprintf('Cramer-Rao bound on estimation of components %2.2f dB\n',mean(-10*log10(mean(crib,2))))

% TENSORBOX v1. 2013