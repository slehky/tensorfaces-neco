% This demo compares algorithms for CPD and nonnegative CPD in
% decomposition of an N-way tensor or k-tensor Y.
% 
% FastALS :  fast ALS algorithm for CP decomposition (CPD)
% fLM:       fast damped Gauss-Newton or Levenberg-Marquardt algorithm
% HALS    :  hierarchical ALS algorithm for nonnegative CPD (NCPD).
% MLS:       multiplicative algorithm for NCPD.
%
% TENSORBOX, 2012
% Phan Anh Huy,  phan@brain.riken.jp
%
clear all
N = 4;   % tensor order
I = ones(1,N) * 30; % tensor size 30 x 30 x ... x 30
R = 10;   % tensor rank  

% Generate a Kruskal tensor from random factor matrices
A = arrayfun(@(x) rand(x,R),I,'uni',0);  % non-negative matrices
Y = ktensor(A(:));  % a Kruskal tensor of rank-R 

% Add Gaussian noise
SNR = 30; % Noise level in dB, inf for noiseless tensor
normY = norm(Y);
if ~isinf(SNR)
    Y = full(Y);
    sig2 = normY./10.^(SNR/20)/sqrt(prod(I));
    Y = Y + sig2 * randn(I);
end
Y = tensor(Y);
% normY = normY/max(Y(:));
% Y = Y/max(Y(:)); 

exectime = [];  % Execution times
Fit = [];       % Fit
MSAE = [];      % mean squared angular error
hline = [];
algs = {};      

%% Cramer-Rao induced bound on factor matrices
try
    if ~isinf(SNR)
        Ca = zeros(R,R,N);
        for n = 1:N
            Ca(:,:,n) = A{n}'*A{n};
        end

        crib = zeros(N,R);
        for n = 1:N
            Cn = Ca(:,:,[n 1:n-1 n+1:N]);
            for r = 1:R
                Cnr = Cn([r 1:r-1 r+1:end],[r 1:r-1 r+1:end],:);
                crib(n,r) = cribNb(Cnr,I(n)) ;
            end
        end
        crib = sig2^2 * crib;
        fprintf('Cramer-Rao Induced bound on factors %2.2f dB\n',mean(-10*log10(mean(crib,2))))
    end
catch
end

%% FastALS for CPD 
opts = cp_fastals;      % get parameters of the FastALS algorithm
opts.init = 'nvec';     % SVD-based initialization ("dtld" is suggested)
opts.linesearch = false;% without line search
opts.printitn = 1;      % print fit every 1 iteration

tic;
[P,output] = cp_fastals(Y,R,opts);
t = toc;

msae = SAE(A,P.U);      % Compute Squared angular error
fit = real(output.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \nExecution time %.3f seconds\n',...
    -10*log10(msae),fit,t);

figure(1);clf; hold on
h = plot(output.Fit(:,1),1-output.Fit(:,2));
xlabel('Iteration')
ylabel('Relative Error')

exectime = [exectime t];        
hline = [hline h];
algs = [algs 'FastALS-CPD'];
Fit = [Fit fit];
MSAE  = [MSAE msae];
    

%% Fast damped Gauss-Newton or Levenberg-Marquardt (fLM) algorithm for CPD
opts = cp_fLMa;     % get parameters of the fLM algorithm
opts.tol = 1e-8;
opts.maxiters = 2000;
opts.init = 'nvec'; % bugs for 'nvecs' for missing data
opts.MaxRecursivelevel = 0; 
opts.maxboost = 0;
opts.printitn = 1;
opts.alsinit = 1;
% opts.tau = 1e-4;
 
tS = tic;
[P,output] = cp_fLMa(Y,R,opts); 
t = toc(tS);

msae = SAE(A,P.U);
fit = real(output.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \nExecution time %.3f seconds\n',...
    -10*log10(msae),fit,t);

figure(1);
h = plot(output.Fit(:,1),1-output.Fit(:,2),'k');
xlabel('Iterations')
ylabel('Relative Error')
axis tight

exectime = [exectime t];
hline = [hline h];
algs = [algs 'fLM-CPD'];
Fit = [Fit fit];
MSAE  = [MSAE msae];

%% HALS for nonnegative CPD
opts = ncp_hals;
opts.init = 'nvec';
opts.printitn = 1;

tic;
[P,output] = ncp_hals(Y,R,opts);
t = toc;

msae = SAE(A,P.U);
fit = real(output.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \nExecution time %.3f seconds\n',...
    -10*log10(msae),fit,t);
figure(1);
h = plot(output.Fit(:,1),1-output.Fit(:,2),'r');
xlabel('Iteration')
ylabel('Relative Error')

exectime = [exectime t];
hline = [hline h];
algs = [algs 'HALS-NCPD'];
Fit = [Fit fit];
MSAE  = [MSAE msae];

%% Multiplicative LS for nonnegative CPD
opts = ncp_mls;
opts.init = 'nvecs';
opts.printitn = 1;

tic;
[P,output] = ncp_mls(Y,R,opts);
t = toc;

msae = SAE(A,P.U);
fit = real(output.Fit(end,2));

fprintf('Mean squared angular error %.3f dB \nFit %.4f \nExecution time %.3f seconds\n',...
    -10*log10(msae),fit,t);
figure(1);
h = plot(output.Fit(:,1),1-output.Fit(:,2),'m');
xlabel('Iteration')
ylabel('Relative Error')

exectime = [exectime t];
hline = [hline h];
algs = [algs 'MLS-NCPD'];
Fit = [Fit fit];
MSAE  = [MSAE msae];

%% Comparse results
figure(1) ; legend(hline,algs);
[foe,fstalg] = min(exectime);
set(gca,'yscale','log')

fprintf('\n')
cprintf('_blue','Algorithm   Exectime (sec) Rel. Error     MSAE (dB)\n');
for k = 1: numel(exectime)
    if k == fstalg
        cprintf('red','%-11s   %.2f         %e    %2.2f \n',...
            algs{k},exectime(k),1-Fit(k), -10*log10(MSAE(k)));
    else
        fprintf('%-11s   %.2f         %e    %2.2f\n',...
            algs{k},exectime(k), 1-Fit(k), -10*log10(MSAE(k)));
    end
end
cprintf('black','')

% TENSORBOX v1. 2012