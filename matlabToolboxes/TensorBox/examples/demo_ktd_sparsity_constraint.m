% This script file is to test the Kronecker tensor decomposition with
% multi approximation terms for image denoising.
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
% Phan Anh Huy (2015)
%
% TENSORBOX
%% Read image
clear all

fname = 'lena_std.tiff';

[pf, fname1, fnameext] =  fileparts(fname);
Y0 = double(imresize((imread(fname)),[256 256]))/255;

% Add noise
SNR = 10;
sig2 = 10^(-SNR/20)*std(double(Y0(:)));
Y = Y0 + sig2 * randn(size(Y0));


%% Generalized KTD with SPARSITY CONSTRAINTS 
% solve the optimization problem
%       min     |Phi(X)|_1   subject to   |Y - Yx|_F^2 < epsilon
%
% or a faster version
%       min     |X_phi|_1   subject to   |Y_phi - Yx_phi|_F^2 < epsilon
%
%  Y_phi = Phi(Y)
%  
%  Phi(Y) : linear transform 
%  epsilon is the noise level,  epsilon = numel(Y)*variance_of_noise
%
% Yx and Yx are transformed using the Kronecker unfolding.

ktd_opts = ktdo; 
ktd_opts.tol = 1e-5;
ktd_opts.abs_tol = true;
ktd_opts.maxiters = 10;
% % ktd_opts.multigroup_updatecycles = 1;
% ktd_opts.multigroups_solver = 'alg';
ktd_opts.step = [4 4];   % shift interval for the convolutive KTD model
ktd_opts.epsilon = sig2; %  sigma: noise level
ktd_opts.autothresh = false;

% KTD supports solvers "OMP", "LASSO" and "BPDN"
Patch_Opt = patch_struct(struct('Size',{{[32 32 1] [nan nan 1]}},...
   'Constraints','sparse','Transform','dct','orthogonal_term',true,'Regularized_par',1e-3,'solver','omp'));

[Yhs,output,~,PatchSize] = ktdo(Y,Patch_Opt,ktd_opts);

% Reconstructed data
Yhs = reshape(Yhs,numel(Y),[]);
Yh = reshape(sum(Yhs,2),size(Y));

% Performance
psnr_ = metrix_mux(Y0*255,Yh*255,'PSNR');
ssim_ = metrix_mux(Y0*255,Yh*255,'SSIM');
fprintf('KTDO, PSNR %.2f, SSIM %.4f\n',psnr_,ssim_)

figure(1);
imagesc([Yh]) 
axis image off
title(sprintf('KTD, Pattern size:%s \n SSIM %.4f',cell2mat(cellfun(@(x) ['(',sprintf('%d,',x(1:end-1)),sprintf('%d)',x(end))] ,PatchSize{1},'uni',0)),ssim_))
pause(1)