% This script runs example 1 in the paper which proposes the AnCU algorithm
% 
% The example approximates a noisy tensor of order-20 by a TT-tensor.
% 
%
% 1 - SVD-truncation
%   1a:  TT-SVD + Rouding
%   1b:  TT-SVD with rank specified
% 2 - ASCU
%   2a- ASCU fits directly to the data
%   2b- ASCU fits directly to the compressed data
% 3 - A2CU
%   3a- A2CU approximates the data
%   3b- A2CU approximates the compressed data
% 4 - A3CU
%   4a- A3CU approximates the data
%   4b- A3CU approximates the compressed data
%
%
% Phan Anh Huy, 2015-2016
%
%% Generate a sinusoid signal of length K = I^2 x 2^d
%
clear all
rng(94520) % for reproducibility

% tensorize the signal to a tensor of order-20 of size 
%  I x 2 x 2 x ... x 2 x I
% number of cores = dx+2
dx = 18;  
I1 = 4 ; % dimenions of the first and last modes
len = I1^2 * 2^dx;K = len; % signal length 

% frequency of sinusoids
freqs = 10;  % frequency of the signal
Fs = ceil(10*max(freqs)); % sampling frequency
phase_ = pi/3; % phase of x(t)
t = (0:len-1); 
x = exp(-t/(len/5)).* sin(2*pi*freqs/Fs * t + phase_); 

% size of tensor tensorized from the signal x(t)
SzY = [I1 2*ones(1,dx) I1];

% TT-rank of the tensor X 
rankR = [1 2*ones(1,dx+1) 1]';

% Generate noise and add it into the signal x(t)
SNR = -20;
y = x;
fprintf('SNR %d dB\n',SNR);
  
if ~isinf(SNR)
    stdY = std(double(y(:)));
    Noise_tens = randn(size(y));
    sig2 = 10.^(-SNR/20)*stdY;
    y=y+sig2*Noise_tens;    
end
Y = reshape(y,SzY);
normY = norm(Y(:));

%% 1a- TT_Truncation with accuracy eps
% \| Y - X \|_F <= tol_svd * \|Y\|_F

% Parameters for the decomposition 
tol = 1e-10;    
maxiters = 1000;

tol_svd = norm(Noise_tens(:))*sig2/normY;
tic;

% approximate Y by a TT-tensor with accuracy tol_svd
Yt1 = tt_tensor(Y,tol_svd,SzY);

% Yt1 may have rank higher than the rank of x
% Rounding TT-tensor to given rank R
Yt1_rd = round(Yt1,tol,rankR);
t1a = toc;

% Relative Approximation error
yx = full(Yt1_rd);
err1a = norm(Y(:) - yx)^2/normY^2;
fig = figure(1);
clf; hold on
clear h
h(1) = plot(1,err1a,'o');
set(h(1),'markersize',18)

[msae_1a,msae2,sae_x1,sae_x2] = SAE({yx},{x'});

%% 1b- TT_Truncation with TT-ranks given
tic;
Yt1b_rd = tt_stensor(Y,tol,SzY,rankR);
% Yt1b_rd = round(Yt1b,tol,rankR);
t1b = toc;
Yt1b_rd = TT_to_TTeMPS(Yt1b_rd);

% Relative Approximation error
yx = reshape(full(Yt1b_rd),[],1);
err1b = norm(Y(:) - yx)^2/normY^2;
fig = figure(1);
h(2) = plot(1,err1b,'o');
set(h(2),'markersize',18)
[msae_1b,msae2,sae_x1,sae_x2] = SAE({yx},{x'});

%% Decomposition using truncated tensor as an initial estimate
% Parameters for the AnCU algorithm , n = 1, 2, 3
opts = ttmps_ascu;
opts.maxiters = maxiters;
opts.tol = tol;
opts.init = Yt1b_rd;
opts.compression = false;
opts.normX = normY;
opts.compression = 0;
opts.noise_level = []; 


tt_algs = {@ttmps_ascu @ttmps_a2cu @ttmps_a3cu};

for ka = 1:numel(tt_algs)
    tt_algorithm = tt_algs{ka};
    tic
    [Yx,out2a] = tt_algorithm(Y,rankR,opts);
    rtime = toc;

    Err_{ka} = 1-out2a.Fit;
    Rtime_(ka) = rtime;
    [msae1,msae2,sae_x1,sae_x2] = SAE({reshape(full(Yx),[],1)},{x'});
    Msae_(ka) = msae1;
    Relerror_(ka) = Err_{ka}(end);
end

%% Part2: Decompose the compressed data  
% Compress the data Y by a TT-tensor with sufficient higher rank
Ytc = tt_tensor(Y,3e-1,SzY);
normYtc = norm(Ytc);
Ytc_mps = TT_to_TTeMPS(Ytc);

% Parameters for the xAnCU algorithms
opts = ttmps_ascu;
opts.maxiters = maxiters;
opts.tol = tol;
opts.init = (Yt1b_rd);
opts.compression = false;
opts.normX = normYtc;
opts.noise_level = [];
 

for ka = 1:numel(tt_algs)
    tt_algorithm = tt_algs{ka};
    tic
    [Yx,out2a] = tt_algorithm(Ytc_mps,rankR,opts);
    rtime = toc;

    Err_cx{ka} = 1-out2a.Fit;
    Rtime_cx(ka) = rtime;
    
    yx = reshape(full(Yx),[],1);
    [msae1,msae2,sae_x1,sae_x2] = SAE({yx},{x'});
    Msae_cx(ka) = msae1;
     
    Relerror_cx(ka) = norm(Y(:) - yx)^2/normY^2;
end
  
%% Compare performances

% Approximation errors 
Relerror = [err1a    err1b     Relerror_     Relerror_cx];
Msae = [msae_1a msae_1b Msae_ Msae_cx];

algs = {'TT-SVD+Rounding' 'TT-SVD_r' 'ASCU' 'A2CU' 'A3CU' ...
    'xASCU' 'xA2CU' 'xA3CU'};

exec_time = [t1a t1b Rtime_ Rtime_cx];

fprintf('%-16s  %7s       %s    %s\n','Algorithm','Error','SAE (dB)','Exec. time')
for k = 1:numel(Relerror)
    fprintf('%-16s    %.5f     %.3f       %2.2f\n',algs{k},Relerror(k),-10*log10(Msae(k)),exec_time(k))
end

%% Plot approximation errors 
clear h 

figure(1);clf
hold on
% h(1) = plot(err1a,'o');
h(1) = plot(err1b,'o');
h(2) = plot(Err_{1},'--');
h(3) = plot(Err_{2},'--');
h(4) = plot(Err_{3},'--');  

set(h,'linewidth',2)
legend(h,{'TT-SVD', 'ASCU'  'A2CU' 'A3CU' })
set(gca,'YScale','log')
set(gca,'XScale','log')

xlabel('Iterations')
ylabel('Relative Error')
set(gca,'fontsize',18)

%%
return