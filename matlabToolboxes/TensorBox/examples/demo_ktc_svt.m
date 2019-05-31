% This example illustrates image completion using the single group
% Kronecker tensor decomposition (KTD) with Singular Value Thresholding.
%
% Data is the color Lena image of size 256 x 256 x 3, but 70% of pixels are
% missing. 
%
% In order to complete missing entries, this degraded image will be
% approximated by KTD as
%
%   Y ~~ A_1 \ox  X_1 + ... + A_R \ox  X_R 
% 
% The decomposition is applied to the data shifted from the original one.
% The final image is reconstructed from all approximate images using the
% median filter.
% 
% Below are some performances PSNR and SSIM
%
% Missing Method    PSNR (dB)  SSIM 
% 70%     LRTC      23.39      0.6161
%         KTC_SVT   28.01      0.8725 (8x8)
%         KTC_SVT   28.4473    0.8764 (8x8,4x4,16x16), shift [4,4] 
%
%
% 95%     LRTC      ~10        ~~0.1
%         KTC_SVT   20.17      0.5294 (8x8)+(16x16), shift [4,4]
%
%
% Phan Anh-Huy (phan@brain.riken.jp)

%% Load image
clear all
Y = imread('lena_std.tiff');
Y = imresize(Y,0.5); % resize image to size of 256 x 256 x3.
Y = im2double(Y);
Y0 = Y;
SzY = size(Y);

% Generate data with missing entries
% W is tensor indicator of 0 and 1. Missing entries in Y are assigned to nan.

missing_ratio = .7;  % missing ratio
missing_ind = rand(SzY(1:2));nopix = prod(SzY(1:2));
[foe,missing_ind] = sort(missing_ind(:));
missing_ind = bsxfun(@plus,missing_ind(1:ceil(missing_ratio*nopix)),0:nopix:2*nopix);

W = true(SzY);
W(missing_ind) = false; % indicator tensor with 0-1 entries
Y(missing_ind) = nan;   % assign nan to missing entries
clear missing_sub missing_ind;

figure(1);imagesc(Y); axis off
title(sprintf('Image with %.0f%% incomplete data',missing_ratio*100))


%% Approximate image by single-group Kronecker tensor decomposition

% Set sizes of patterns X
% Ix is an array of G rows corresponding to G groups of patterns,
%       each row indicates size of patterns in the same group.
%

if missing_ratio < .95
    Ix = [8 8 1];% When missing ratio is 70%,
    maxiters = 1000;
else
    Ix = [8 8 1
        16 16 1];
    maxiters = 3000;
end
Nogroups = size(Ix,1);

Yshift = [];

%% Run single group KTC-SVT for each group of patterns

for kg = 1:Nogroups
     
    Ix_g = Ix(kg,:);    % pattern size
    step = min(4,Ix_g(1:2)/2); % shift image left/right/up/down
 
    %% Singular value thresholding for single group KTC
    opts = ktc_svt;
    opts.shift_type = 'sequential';
    opts.maxiters = maxiters;
    opts.step = step;
    [Yh,output,Yshift_g] = ktc_svt(Y,Ix_g,opts);
    
    %% Concatenate estimate images for different groups of patterns
    Yshift = cat(4,Yshift,Yshift_g);
end


%% Reconstruct the final image
Yh = median(Yshift,4);
Yh(W) = Y(W);
Yh = min(max(0,Yh),1);

% Assess performance PSNR and SSIM
psnr_ = 20*log10(sqrt(numel(Y0))/norm(Yh(:) - Y0(:)));
ssim_ = metrix_mux(Y0*255,Yh*255,'SSIM');

fprintf('PSNR %.2f, SSIM %.4f\n',psnr_,ssim_)

% Visualize the estimated image
figure(2);clf
imagesc(Yh);axis off
title(sprintf('PSNR %.2f, SSIM %.4f\n',psnr_,ssim_))

return