% TT-Decomposition with a given error bound 
%      |Y - Yx|_F^2 <= error = noise level
%
% Phan Anh Huy, 2015-2016
%
%%

% A signal corrupted by noised is tensorized to a tensor of order-20 of size 
%  I x 2 x 2 x ... x 2 x I
%
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
rank_x = [1 2*ones(1,dx+1) 1]';

% Generate noise and add it into the signal x(t)
SNR = 0;
y = x;
fprintf('SNR %d dB\n',SNR);
  
if ~isinf(SNR)
    stdY = std(double(y(:)));
    Noise_tens = randn(size(y));
    noise_level = 10.^(-SNR/20)*stdY;
    y=y+noise_level*Noise_tens;    
end
Y = reshape(y,SzY);
normY = norm(Y(:));


%% Decompose the tensor Y with known accuracy level 
% if not given, the noise level can be estimated from the signal
error_bound = sqrt(numel(Y))*noise_level; 
% error_bound = noise_level * norm(Noise_tens); %

% Using the TT-SVD algorithm 
tol_svd = error_bound/normY;
tic;
Ytsvd = tt_tensor(Y,tol_svd);
t_svd = toc;

% SAE 
x_tc = full(Ytsvd);
sae_tc = SAE({x_tc(:)},{x(:)});
fprintf('SAE of X: %s \n',sprintf('%.2f dB, ',-10*log10(sae_tc)))

%% Decompose the tensor Y using AnCU
rankR = [];
opts = ttmps_ascu;
opts.compression = false;
opts.maxiters = 100; %maximum number of iterations;
opts.tol = 1e-6;
opts.core_step = 1;
opts.exacterrorbound = true;
 
opts.printitn = 1;
% opts.normX = normY;
opts.rankadjust = 1;

t_acu = zeros(1,3);
Out_acu = cell(1,3);
sae_acu = zeros(1,3);
x_ts= zeros(numel(x),3);
TT_acu_est = cell(3,1);
tt_algs = {@ttmps_ascu @ttmps_a2cu @ttmps_a3cu};

%%
for kalg = 1:3
    C = 1.5;
    opts.core_step = 2;

    opts.init = TT_to_TTeMPS(Ytsvd);
    %opts.init = 'nvecs';
    accuracy = C* error_bound^2;
    opts.noise_level = accuracy;
    tic;
    [Xt,out2a] = tt_algs{kalg}(Y,rankR,opts);
    t_acu(kalg) = toc;
    
    C = 1.;
    accuracy = C* error_bound^2;
    opts.noise_level = accuracy;
    
    %%
    while 1
        
        Xt = tt_kickrank(Y,Xt);
   
        %%
%         for ki = 1:3
            opts.init = Xt;
            [Xt,out2a] = tt_algs{kalg}(Y,rankR,opts);
%         end

        % SAE
        TT_acu_est{kalg} = Xt;
        xf = reshape(full(Xt),[],1);
        x_ts(:,kalg) = xf;
        %%
        if norm(Y(:)-xf)^2 <= accuracy*1.0001
            break
        end
    end
    
    %%
   
    sae_ts = SAE({x_ts(:,kalg)},{x(:)});
    fprintf('Alg%d , SAE of X: %s \n',kalg,sprintf('%.2f dB, ',-10*log10(sae_ts)))
    
    sae_acu(kalg) = sae_ts;
    Out_acu{kalg} = out2a;
end

%%

% Approximation errors 
Relerror = [norm(Y(:)-x_tc)/normY  cellfun(@(x) x.Fit(end),Out_acu) ];
Msae = [sae_tc sae_acu];

algs = {'TT-SVD' 'ASCU' 'A2CU' 'A3CU'};

fprintf('%-16s  %7s       %s   \n','Algorithm','Error','SAE (dB)')
for k = 1:numel(Relerror)
    fprintf('%-16s    %.5f     %.3f      \n',algs{k},Relerror(k),-10*log10(Msae(k)))
end

%%
fprintf('----------------------------------------\n')
fprintf('(true)-Rank      %s \n', sprintf('%d-',rank_x))
fprintf('Rank_tt          %s \n', sprintf('%d-',rank(Ytsvd)))
fprintf('Rank_ASCU        %s \n', sprintf('%d-',TT_acu_est{1}.rank))
fprintf('Rank_ADCU        %s \n', sprintf('%d-',TT_acu_est{2}.rank))
fprintf('Rank_ATCU        %s \n', sprintf('%d-',TT_acu_est{3}.rank))
fprintf('----------------------------------------\n')