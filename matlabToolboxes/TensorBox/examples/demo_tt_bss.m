% This example shows an application of TT-decomposition for BSS 
% Separation of signals which have low-rank Tensor train representation 
% from only one observed signal.
%
% The method applies two tensorizations : Toeplitz and Reshaping
%
% PHAN ANH-HUY
% TENSORBOX 

%%
clear all

% Signal y is a mixture of three component signals x1, x2 and x3 of length
% K = 1086
%
% x1 = sin(2*pi*w(1)*t + phase_(1));
% x2 = t.*sin(2*pi*w(2)*t + phase_(2));
% x3 = exp(-t/(K/20)).*sin(2*pi*w(3)* t + phase_(3)));
%
%
R = 3;       % number of signal components
freqs = [7 8 9]; % frequency of sinusoids
Fs = ceil(10*max(freqs)); % sampling frequency

% The task is to retrieve the component signals x1, x2 and x3 from only one
% mixture y.
%
% The component signals, x_r, can be represented by TT-tensor of low rank
% (2,2,...,2) or (4,4,...,4)
% The Toeplitz tensors of the signals also have low-rank TT
% representations.
% 
% By exploiting the low-rank structure of the signal x_r(t),
% we can tensorize the signal to a higher order tensor 
% and decompose the tensor to retrieve the latent component signals.


% In order to achieve this, we first construct a Toeplitz tensor/matrix
% from the mixture y. 
% The Toeplitz tensor/matrix has order of Toeplitz_order
% and size I*2^dx x 2^dx x ... x 2^dx x I*2^dx

toeplitz_order = 3; % Toeplitz order: 2: Toeplitz matrix, 3,4 ...: Toeplitz tensor
dx = 6;
I1 = 2*R + 2;
toeplitz_size = [I1*2^dx  2^dx*ones(1,toeplitz_order-2) I1*2^dx];
fToeplitz = toeplitzation(toeplitz_size); % linear operator to construct a Toeplitz tensor

% Then reshape the Toeplitz tensor to give a tensor of order-20 
% (dx*toeplitz_order + 2)
% Tensor size after the 2nd tensorization
sz_x = [I1 2*ones(1,dx*toeplitz_order) I1];

% Length of the signal
len = (2*I1+toeplitz_order-2) * 2^dx- toeplitz_order + 1; % signal length

% Signal and mixture
t = (0:len-1);
phase_ = pi./R*(0:R-1);

% Generate components signals x(t)
x = zeros(R,len);
for k = 1:R
    x(k,:) = sin(2*pi*freqs(k)'/Fs * t + phase_(k));
end
% x2
x(R-1,:) = x(R-1,:) .* t;
% x3
x(R,:) = exp(-t/(len/20)).*x(R,:);  

% Scaling component signals
ssx = sqrt(sum(x.^2,2));
x = bsxfun(@rdivide,x,ssx/ssx(1));

% Display the component signals x(t)
figure(1);
clf
plot(x');
title('Component Signals x_n(t)')

%% Check TT-rank of component signals x_r after two tensorizations
fprintf('Check TT-rank of component signals')
clear rank_x;
for r = 1:R
    %% toeplitzation
    xtp = fToeplitz.A(x(r,:)');
    xtt_r{r} = tt_tensor(reshape(xtp,sz_x),1e-6,sz_x);
    fprintf('Component %d, Rank (%s)\n',r,sprintf('%d,',rank(xtt_r{r})'))
    rank_x(r,:) = rank(xtt_r{r})';
end

%% Generate a noisy signal
y0 = sum(x);
 
% Add noise to the mixture y
SNR = 10; % dB
y = y0;
if ~isinf(SNR)
    stdY = std(double(y(:)));
    Noise_tens=randn(size(y));
    sig2 = 10.^(-SNR/20)*stdY;
    y=y+sig2*Noise_tens;
end

% Display the component signals x(t)
figure(2);
clf
plot(y');
title('Observed Signal y(t)')

%% Main Process to retrieve the component signals
opts = [];
[A,xhat,x_tt,error] = tt_bss(y,R,sz_x,rank_x,toeplitz_order,Fs,opts);

%% Scaling and reorder the estimated signals 
xhat = xhat';
xhat = bsxfun(@rdivide,xhat,sqrt(sum(xhat.^2,2)));
xn = bsxfun(@rdivide,x,sqrt(sum(x.^2,2)));

cc = xhat*xn';
[foe,cidx] = max(abs(cc));
[cidx2,ii,jj] = unique(cidx);
cidx = [cidx(sort(ii)) setdiff(1:R,cidx2)];

idx = sub2ind(size(cc),cidx,1:size(cc,1));
xhat = bsxfun(@times,xhat(cidx,:),sign(cc(idx))'.*sqrt(sum(x(cidx,:).^2,2)));
 
% Assess square angular errors 
[msae1,msae2,sae_a,sae2] = SAE({xhat'},{xn'});

fprintf('Mean Square angular error of the estimated signals(dB)\n')
fprintf('%s \n',sprintf('%.2f dB, ',-10*log10(sae_a)))


%% Visualize the estimated signals 
close all
fig = figure(1);
clf
subplot(311)
h1= plot(y,'-');
set(h1,'linewidth',3)
axis tight
% legend([h1(1) h2(1)],{'True' 'Estimate'})
set(gca,'fontsize',18)
xlabel('Samples')
% ylabel('Relative Intensity')
title('Mixture')

subplot(312)
h1= plot(x','-');
set(h1,'linewidth',3)
axis tight
% legend([h1(1) h2(1)],{'True' 'Estimate'})
set(gca,'fontsize',18)

xlabel('Samples')
% ylabel('Relative Intensity')
title('Source signals')

subplot(313)
h2 = plot(xhat','-'); hold on
set(h2,'linewidth',3)
axis tight
set(gca,'fontsize',18)
xlabel('Samples')
% ylabel('Relative Intensity')
title('Estimated signals')