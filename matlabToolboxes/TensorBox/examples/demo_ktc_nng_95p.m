% This example illustrates image completion using the Kronecker tensor
% decomposition (KTD) with nonnegativity constraints
%
% Data is the color Lena image of size 512 x 512 x 3 (or 256 x 256 x 3),
% but 70% of pixels are missing.
%
% In order to complete missing entries, this degraded image will be
% approximated by KTD as
%
%   Y ~~ A_1 \ox  X_1 + ... + A_R \ox  X_R
%
% where \ox denotes the Kronecker product between two tensors A_r and X_r.
% In this example, X_r are of size 8x8x1 for single-group KTD or (8x8x1)
% and (4x4x1) for multigroup KTD.
%
% The performance of the completion (PSNR) will be improved 4-5 dB when the
% KTD is performed on the data which are shifted left/right/up/down
% from the original data in a range of [-2 2] or [-4 4].
%
% The final image is reconstructed from all approximate images using the
% median filter.
%
% Below are some results for missing ratios of 70% and 95%.
% _________________________________________________________________________
%% Image size 256 x 256 x 3, 70\% of missing entries
% _________________________________________________________________________
% Conf PSNR (dB)  SSIM    Group size   Multigroup     Shift    Shift type
% -------------------------------------------------------------------------
% 1s   22.70,     0.5250  (8x8)
% 2s   26.33,     0.8121  (8x8)                      (4,4)    sequential
% 3s   27.18,     0.8432  (8x8)                      (4,4)    augmented
%
% 1m   23.54,     0.5869  (4x4)+(8x8)  simultaneous
% 2m   23.10,     0.5634  (4x4)+(8x8)  sequential
% 3m   26.99,     0.8254  (4x4)+(8x8)  simultaneous  (4,4)    sequential
% 4m   26.48,     0.8026  (4x4)+(8x8)  sequen        (4,4)    sequential
% 5m   27.44,     0.8435  (4x4)+(8x8)  simultaneous  (4,4)    augmented
% 6m   27.24,     0.8312  (4x4)+(8x8)  sequential    (4,4)    augmented
% _________________________________________________________________________
%
%% Image size 512 x 512 x 3, 70% of missing entries
% _________________________________________________________________________
% Conf PSNR (dB)  SSIM    Group size   Multigroup    Shift    Shift type
% -------------------------------------------------------------------------
% 3s   28.64,     0.8460  (8x8)                      (4,4)    augmented
%
% 5m   29.23,     0.8509  (4x4)+(8x8)  simultaneous  (4,4)    augmented
% 6m   28.75,     0.8311  (4x4)+(8x8)  sequential    (4,4)    augmented
% _________________________________________________________________________
%
%% Image size 512 x 512 x 3, 95\% missing entries
% _________________________________________________________________________
% Conf PSNR (dB)  SSIM    Group size    Multigroup    Shift    Shift type
% -------------------------------------------------------------------------
% 1m   15.22,     0.2555  (8x8)
% 3m   18.52,     0.5373  (8x8)                      (4,4)    augmented
%
% 5m   22.90,     0.6827  (8x8)+(16x16) sequential   (8,8)    augmented
% 6m   21.62,     0.6397  (8x8)+(16x16) sequential   (8,8)    augmented
% _________________________________________________________________________
%
%% Image size 256 x 256 x 3, 95\% missing entries
% _________________________________________________________________________
% Conf PSNR (dB)  SSIM    Group size    Multigroup   Shift    Shift type
% -------------------------------------------------------------------------
% 2s   17.36,     0.4330  (8x8)                      (4,4)    sequential
% 3s   17.50,     0.4930  (8x8)                      (4,4)    augmented
%
% 5m   21.17,     0.6163  (8x8)+(16x16) sequential   (8,8)    augmented
% 6m   20.02,     0.5648  (8x8)+(16x16) sequential   (8,8)    augmented
% _________________________________________________________________________
%
%% Image size 256 x 256 x 3, 97\% missing entries
% _________________________________________________________________________
% Conf PSNR (dB)  SSIM    Group size    Multigroup   Shift    Shift type
% -------------------------------------------------------------------------
% 5m   18.8447    0.5426  (8x8)+(16x16) sequential   (4,4)    augmented
% _________________________________________________________________________
%
% Completion of image of size 512x512x3 achived higher performance than
% that of size 256x256x3.
%
% Decomposition for augmented data from shiftings such as "3s", "5m" or
% "6m" is recommended.
% The shift parameter "step" should be half of pattern sizes. For example,
% step = [4 4] for pattern size of [8 x 8]. In this case, the data is
% augmented by 81 (=9x9) times.
% 
% When missing ratio is relatively high, e.g., 95\%, the pattern size
% should be larger, e.g., 16x16.
%
% See also: ex_ktc_svt.m
%
% This file is a part of the TENSORBOX (2014).
% Phan Anh-Huy (phan@brain.riken.jp)

%% Load image
clear all
% Y = imread('barbara.jpeg'); %
Y = imread('lena_std.tiff');
Y = imresize(Y,0.5); % 256 x 256 x 3 
Y = im2double(Y);
Y0 = Y;
I = size(Y);

% Generate data with 95% of missing pixels
% W is a tensor indicator of 0 and 1. Missing entries are set to nan.

missing_ratio = .95; nopix = prod(I(1:2));
missing_ind = rand(I(1:2));[foe,missing_ind] = sort(missing_ind(:));
missing_ind = bsxfun(@plus,missing_ind(1:ceil(missing_ratio*nopix)),0:nopix:2*nopix);

W = true(I);W(missing_ind) = false; % indicator tensor with 0-1 entries
Y(missing_ind) = nan;   % assign nan to missing entries
clear missing_ind;

figure(1);imagesc(Y); axis off
title(sprintf('Image with %.0f%% incomplete data',missing_ratio*100))

%% Setting configuration for KTC
 
% Get default parameters
opts = ktc_nng;
opts.verbose = 1;
opts.normA = 2e-2;
opts.smoothA = 0;
opts.maxmse = 30;
opts.maxiters = 200;

% Set sizes for pattern X
Ix = [8 8 1; 16 16 1];
opts.step = [4 4];  % shift interval
opts.shift_type = 'augmented';
opts.multigroup_type = 'simultaneous';


% no need to change the following parameters
Ia = bsxfun(@rdivide,I,Ix);     % size of patterns A
P = min([prod(Ia,2),prod(Ix,2)],[],2); % number of tensors in each group.

%% KTC

Yh = ktc_nng(Y,Ix,P,opts);

% Evaluate performance
Yh(W) = Y(W);
Yh = min(max(0,Yh),1);

% Assess performance PSNR and SSIM
psnr_ = 20*log10(sqrt(numel(Y0))/norm(Yh(:) - Y0(:)));
ssim_ = metrix_mux(Y0*255,Yh*255,'SSIM');

fprintf('PSNR %.2f, SSIM %.4f\n',psnr_,ssim_)

% Visualize the esimated image
figure(2);clf
imagesc(Yh);axis off
title(sprintf('PSNR %.2f, SSIM %.4f\n',psnr_,ssim_))

return