% An example compares TT-truncation and TT-AnCU
 
% Generate signal and a tensor of order (dx+2)
% and size I1 x 2 x ... x 2 x I1
clear all;

dx = 20; % tensor order will be of dx+2
I1 = 2;  
len = I1^2 * 2^dx;  % length of signal

% Sampling frequency 
Fs = 5000;

sz_x = [I1 2*ones(1,dx) I1];

phase_ = pi/3; % phase of x(t)
freqs = 10;
t = (0:len-1);
x = exp(-t/(len/5)).* sin(2*pi*freqs/Fs * t + phase_);
                 
% Plot the signal x
figure(1);clf
plot(x)

% Tensor of this signal has rank
rankx = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1]';
 
y = x; 

% Add noise into the signal x
SNR = 0;
fprintf('SNR %d dB\n',SNR);
if ~isinf(SNR)
    stdY = std(double(y(:)));
    Noise_tens=randn(size(y));
    noise_level = 10.^(-SNR/20)*stdY;
    y=y+noise_level*Noise_tens;
    err_norm = norm(Noise_tens);
    clear Noise_tens;
end 
Y = reshape(y,sz_x);
normY = norm(y);
 
% parameters for the TT decomposition
tol = 1e-10;
maxiters = 1000;
  
%% Decompose using TT-SVD with known rank
error_bound = sqrt(numel(Y))*noise_level; 
% error_bound = noise_level * norm(Noise_tens); %

% Using the TT-SVD algorithm 
tol_svd = error_bound/normY;
tic;
Ytsvd_3 = tt_stensor(Y,tol_svd/3,sz_x,rankx(:));
t_svd3 = toc;

% SAE 
x_tc3 = full(Ytsvd_3);
sae_tc3 = SAE({x_tc3(:)},{x(:)});
fprintf('SAE of X: %s \n',sprintf('%.2f dB, ',-10*log10(sae_tc3)))


%% Decompose the tensor Y using AMCU 
rankR = [];
opts = ttmps_ascu;
opts.compression = 0;
opts.maxiters = 200; %maximum number of iterations;
opts.tol = 1e-10;
opts.exacterrorbound = true;

opts.core_step = 2;

opts.init = Ytsvd_3;   % forex  2
opts.init = TT_to_TTeMPS(opts.init);


opts.printitn = 1;
% opts.normX = normY;
opts.rankadjust = 1;

t_acu = zeros(1,3);
Out_acu = cell(1,3);
sae_acu = zeros(1,3);
x_ts= zeros(numel(x),3);
TT_acu_est = cell(3,1);
for kalg = 1:3
     
    tic;
    switch kalg
        case 1
            [tt_acu,out2a] = ttmps_ascu(Y,rankR,opts);
            
        case 2
            [tt_acu,out2a] = ttmps_a2cu(Y,rankR,opts);
            
        case 3
            [tt_acu,out2a] = ttmps_a3cu(Y,rankR,opts);
    end
    t_acu(kalg) = toc;
    

    % SAE
    TT_acu_est{kalg} = tt_acu;
    x_ts(:,kalg) = reshape(full(tt_acu),[],1);
    sae_ts = SAE({x_ts(:,kalg)},{x(:)});
    fprintf('Alg%d , SAE of X: %s \n',kalg,sprintf('%.2f dB, ',-10*log10(sae_ts)))
    
    sae_acu(kalg) = sae_ts;
    Out_acu{kalg} = out2a;
end
   
%% Compare signal and its estimated
kalg = 1;
fig = figure(1);
clf
% h = plot([xhat At(y)]);
hold on
clear h
h(1) = plot(y);
h(2) = plot(x);

k = 0;
while k<8
    k = k+1;
    ix = len/8*(k-1)+1:len/8*k;
    h(2+k) = plot(ix,x_ts(ix,kalg));
end

%%
clear ix 
set(h(2),'linestyle','-.')
set(h(2),'linewidth',2)
set(h(3:2:end),'linestyle','-')
set(h(4:2:end),'linestyle','-','linewidth',2)

set(h(3:2:end),'color',[0.4940 0.1840 0.5560]);
set(h(4:2:end),'color','r');%[0.4660 0.6740 0.1880]);

 
set(gca,'fontsize',18)
legend(h,{'Noisy' 'Source'  'Truncation' 'Alternating' },'location','best')
drawnow
axis tight
xlabel('Samples')
ylabel('Intensity')
 
% %%
% figname = sprintf('fig_tt_denoising_sr%dB_ex%d',SNR,ex);
% saveas(fig,[figname '.fig'],'fig')
% print(fig,'-depsc',[figname '.eps'])
%%
fprintf('           SAE (dB)     Time\n')
fprintf('TTSVD1 : %+11s %2.4f\n',sprintf('%.2f , ',-10*log10(sae_tc3)),t_svd3)
fprintf('ASCU   : %+11s %2.4f\n',sprintf('%.2f , ',-10*log10(sae_acu(1))),t_acu(1))
fprintf('A2CU   : %+11s %2.4f\n',sprintf('%.2f , ',-10*log10(sae_acu(2))),t_acu(2))
fprintf('A3CU   : %+11s %2.4f\n',sprintf('%.2f , ',-10*log10(sae_acu(3))),t_acu(3))
fprintf('----------------------------------\n')


return