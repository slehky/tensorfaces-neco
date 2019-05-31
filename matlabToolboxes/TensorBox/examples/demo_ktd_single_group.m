% This script file is to test the Kronecker tensor decomposition with
% single approximation term for image denoising.
%
%    Y = A1 o X1 + A2 o X2 + ... + AR o XR
%
% Patch structures, including patch sizes, type of constraints and optimization problem, 
% are set using the subroutine "patch_struct".
% 
% For example, KTD approximates the data by two patterns of size (32x32x1)
% and (8x8x3), respectively, with sparsity constraint on DCT domain 
%
%       Patch_Opt = patch_struct;
%       Patch_Opt.Size = {[32 32 1] [8 8 3]};
%       Patch_Opt.Constraints = 'sparse';
%       Patch_Opt.Transform = 'dct';
%
% An approximation term is composed by 2 patterns of size [32 32 1] x [8 8 3]
% and with rank-minimal constraint
%
%       Patch_Opt = patch_struct;
%       Patch_Opt.Size = {[32 32 1] [8 8 3]};
%       Patch_Opt.Constraints = 'lowrank';
%
% An approximation term is constructed from 50 (rank) patterns of size [4 4 1] x [8 8
% 1] x [8 8 3] 
%
%       Patch_Opt = patch_struct;
%       Patch_Opt.Size = {[4 4 1] [8 8 1] [8 8 3]};
%       Patch_Opt.Constraints = 'lowrank';
%       Patch_Opt.NoComps = 50;
%
% Phan Anh Huy
%
% TENSORBOX 2018
%% Read image
clear all

fname = 'lena_std.tiff';

[pf, fname1, fnameext] =  fileparts(fname);
Y0 = double(imresize((imread(fname)),[256 256]))/255;

% Add noise
SNR = 10;
sig2 = 10^(-SNR/20)*std(double(Y0(:)));
Y = Y0 + sig2 * randn(size(Y0));

%% Example 1: KTD approximates the image by a single group of 10 patterns of size [32x32x1] x [8 8 3]
% This example boils down to low-rank matrix factorization
%
ktd_opts = ktdo;
ktd_opts.tol = 1e-5;
ktd_opts.abs_tol = true;
ktd_opts.maxiters = 10;

Patch_Opt = patch_struct(struct('Size',{{[32 32 1]  [nan nan 1]}},...
    'Constraints','lowrank','Regularized_par',1,'NoComps',[10],'solver','nc'));

[Yh,output,~,PatchSize] = ktdo(Y,Patch_Opt,ktd_opts);

% Performance
psnr_ = metrix_mux(Y0*255,Yh*255,'PSNR');
ssim_ = metrix_mux(Y0*255,Yh*255,'SSIM');
fprintf('KTDO, PSNR %.2f, SSIM %.4f\n',psnr_,ssim_)

figure(1);
imagesc([Yh]) 
axis image off
title(sprintf('KTD, Pattern size:%s \n SSIM %.4f',cell2mat(cellfun(@(x) ['(',sprintf('%d,',x(1:end-1)),sprintf('%d)',x(end))] ,PatchSize{1},'uni',0)),ssim_))
pause(1)


%% Example 2: Convolutive-like KTD approximates the image by 10 patterns of size [32x32x1] x [8 8 3]
% KTD first augmentes the data by 25 copies of itself but shift in a small
% region: [-2,2].
% This example boils down to low-rank matrix factorization
%
ktd_opts = ktdo;
ktd_opts.tol = 1e-5;
ktd_opts.abs_tol = true;
ktd_opts.maxiters = 10;
ktd_opts.step = [2 2];

Patch_Opt = patch_struct(struct('Size',{{[32 32 1]  [nan nan 1]}},...
    'Constraints','lowrank','Regularized_par',1,'NoComps',[10],'solver','nc'));

[Yh_cn,output,~,PatchSize] = ktdo(Y,Patch_Opt,ktd_opts);

% Performance
psnr_cn = metrix_mux(Y0*255,Yh_cn*255,'PSNR');
ssim_cn = metrix_mux(Y0*255,Yh_cn*255,'SSIM');
fprintf('KTDO, PSNR %.2f, SSIM %.4f\n',psnr_cn,ssim_cn)

figure(2);
imagesc(Yh_cn) 
axis image off
title(sprintf('KTD, Pattern size:%s \n SSIM %.4f',cell2mat(cellfun(@(x) ['(',sprintf('%d,',x(1:end-1)),sprintf('%d)',x(end))] ,PatchSize{1},'uni',0)),ssim_cn))
pause(1)

axis image off
pause(1)
 
%% Example 3: Convolutive-like KTD approximates the image by 1000 patterns of size [2 2 1]x[2 2 1]x[2 2 1]x[2 2 1]x[8 8 3]
% KTD first augmentes the data by 25 copies of itself but shift in a small
% region: [-2,2].
% This example boils down to low-rank approximation of tensors of size
% 25 x 4 x 4 x 4 x 4 x 4 x 192
%
ktd_opts = ktdo;
ktd_opts.tol = 1e-5;
ktd_opts.abs_tol = true;
ktd_opts.maxiters = 10;
ktd_opts.step = [2 2];

Patch_Opt = patch_struct(struct('Size',{{[1 1 1] [2 2 1] [2 2 1] [2 2 1] [2 2 1] [2 2 1] [nan nan 3]}},...
    'Constraints','lowrank','Regularized_par',1,'NoComps',1000,'solver','nc'));

[Yh_cn6,output,~,PatchSize] = ktdo(Y,Patch_Opt,ktd_opts);

% Performance
psnr_cn6 = metrix_mux(Y0*255,Yh_cn6*255,'PSNR');
ssim_cn6 = metrix_mux(Y0*255,Yh_cn6*255,'SSIM');
fprintf('KTDO, PSNR %.2f, SSIM %.4f\n',psnr_cn6,ssim_cn6)

figure(3);
imagesc(Yh_cn6) 
axis image off
title(sprintf('KTD, Pattern size:%s \n SSIM %.4f',cell2mat(cellfun(@(x) ['(',sprintf('%d,',x(1:end-1)),sprintf('%d)',x(end))] ,PatchSize{1},'uni',0)),ssim_cn6))
pause(1)

%% Summarize results

fprintf('KTD    PSNR(dB)   SSIM\n')
fprintf('Ex1    %2.2f      %.4f\n',psnr_,ssim_)
fprintf('Ex2    %2.2f      %.4f\n',psnr_cn,ssim_cn)
fprintf('Ex3    %2.2f      %.4f\n',psnr_cn6,ssim_cn6)
