%%
clear all
input_snr = 10;

% Original image
im_name = 'lena_std.tiff';% 'lena_std.tiff' 'tiffany.tiff' 'house.tiff'
[~,im_name_,im_ext] = fileparts(im_name);
Y0=im2double(imread(im_name));

sz_im = [256 256];
Y0 = imresize(Y0,sz_im);
SzY = size(Y0);

% Add noise to the image
sigma_noise = 10^(-input_snr/20)*std(Y0(:));
Y = Y0 + randn(SzY)*sigma_noise;

% Display the noisy image
figure(2)
imagesc(Y);
colormap gray;
hold on;
axis off
axis image

%% Denoising method constructs tensors from patches of size d x d x 3 and their neighbours
%
blk_size = [8 8]; % patch (block) size
neighb_range = 2; % defin the area +/-2 around the observed block 
shiftstep = 2;    % 
get_rank = true;  % true to get the estimated rank of each tensor decomposition
colorspace = 'opp';
decomposition_method =  'ttmps_ascu'; % tensor decomposition method ASCU for TT
                                      % Other methods include ttmps_adcu,
                                      % cpdepc, brtf, tt_truncation, tucker
 
% Run the decomposition 
Yhm = tt_image_denoising_neighbour_(Y,blk_size,neighb_range,sigma_noise,decomposition_method,shiftstep,get_rank,colorspace,[],im_name_);

%% Assess performance of denoising 
Perf_  = {};
for metrix = {'MSE' 'PSNR' 'SSIM'}
    perf_ = metrix_mux(Y0*255,Yhm *255,metrix{1});
    Perf_ = [Perf_ ; metrix {perf_}];
end
fprintf('Denoising result\n')
Perf_

%%
fig = figure(1);
clf
imagesc(Yhm)
axis image
axis off

Perf_
 
%%
imedge = edge(rgb2gray(Y0),'canny');
% BW2 = bwperim(imedge,4);
% imedge = imdilate(BW2, strel('disk',1));

imf = imfuse(double(imedge)*max(blkrank_(:)),blkrank_*2,'blend','Scaling','joint');
imf = double(imf)/max(double(imf(:))) * max(blkrank_(:));
fig = figure(2);
clf
imagesc(imf);
axis image; 
axis off;

colorbar

figname = sprintf('fig_%s_blkrank_epc',im_name_);
saveas(fig,[figname '.fig'],'fig')
print(fig,'-depsc',[figname '.eps'])

%%

return
