%DEMO_SDMM  Example of use of the sdmm solver 
%
%   We present an example of the solver through an image
%   denoising problem.
%   The problem can be expressed as this
%
%       argmin_x,y,z ||x-b||_2^2 + tau1*||y||_TV + tau2 * ||H(z)||_1 such that  x = y = Hz
%
%
%   Where b is the degraded image, tau_1 and tau_2 two real positive constant and H a linear operator on x.
%   H is a wavelet operator. We set:
%
%    g_1(x)=||x||_{TV}
%     We define the prox of g_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||z||_TV
%
%
%    g_2(x)=||H(x)||_1
%     We define the prox of g_2 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||H(z)||_1
%
%
%    f(x)=||x-b||_2^2
%     We define the gradient as: 
%
%        grad_f(x) = 2 * (x-b)
%
%
%   Results
%   -------
%
%   Figure 1: Original image
%
%      This figure shows the original image (The cameraman). 
%
%   Figure 2: Depleted image
%
%      This figure shows the image after addition of the noise
%
%   Figure 3: Reconstruted image
%
%      This figure shows the reconstructed image thanks to the algorithm.
%   
%   The rwt toolbox is needed to run this demo.
%
%   References:
%     P. Combettes and J. Pesquet. Proximal splitting methods in signal
%     processing. Fixed-Point Algorithms for Inverse Problems in Science and
%     Engineering, pages 185-212, 2011.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_sdmm.php

% Copyright (C) 2012-2013 Nathanael Perraudin.
% This file is part of LTFAT version 1.1.97
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Author: Nathanael Perraudin
% Date: November 2012


%% Initialisation

%clear all;
close all;
clc;



% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();



% setting different parameter for the simulation

tau1 = 1e0; %parameter for the problem
tau2 = 1e0; %parameter for the problem

% Original image
im_original=im2double(imread('misc/cameraman.tif'));
im_original=imresize(im_original,[256 256]);

% Displaying original image
figure(1);
imagesc(im_original);
colormap gray;
hold on;
title('Original image');
axis off          
axis image        


% Depleted image
input_snr = 20;
b = im_original;
sigma_noise = 10^(-input_snr/20)*std(b(:));
b = b + randn(size(b))*sigma_noise;


% Displaying depleted image
figure(2);
imagesc(b);
colormap gray;
hold on;
title('Depleted image');
axis off          
axis image        

% setting the function f1 

% for the TV norm
param2.verbose=1;
param2.maxit=100;
param2.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

g1.prox=@(x, T) prox_tv(x, T*tau1, param2);
g1.eval=@(x) tau1*tv_norm(x);   
g1.x0=b;
g1.L=@(x) x;
g1.Lt=@(x) x;

% for the nuclear norm
param2.verbose=1;
param2.maxit=100;
param2.useGPU = 0;   % Use GPU for the TV prox operator.

g2.prox=@(x, T) prox_nuclearnorm(x, T*tau2, param2);
g2.eval=@(x) tau2*norm_nuclear(x);   
g2.x0=b;
g2.L=@(x) x;
g2.Lt=@(x) x;

% for the projection
g3.prox=@(x, T) (2*T*b+x)/(2*T+1);
g3.eval=@(x) eps;  
g3.x0=b;
g3.L=@(x) x;
g3.Lt=@(x) x;

% Parameter for the sum of function: F
F={g2,g1, g3};
param4.max_iter=10;

% solving the problem
sol=sdmm(F,param4);


% displaying the result
figure(3);
imagesc(sol);
colormap gray;
hold on;
title('Reconstructed image');
axis off         
axis image        

close_unlocbox();

