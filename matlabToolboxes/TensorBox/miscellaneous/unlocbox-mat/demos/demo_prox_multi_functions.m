%DEMO_PROX_MULTI_FUNCTIONS Demonstration of the proximal operator of a sum of function
%   
%   In this example we solve a image denoising problem. We remember the
%   reader that solving a proximal operator can be view like denoising a
%   signal. The function used is prox_sumg which compute the proximal
%   operator of a sum of function.
%
%   The problem can be expressed as this
%
%        argmin ||x-y||^2 + tau1*||x||_TV + tau2 * ||H(x)||_1
%
%  
%   Where z is the degraded image.
%
%   H is a linear operator projecting the signal in a sparse
%   representation. Here we worked with wavelet. 
%
%   Warning! Note that this demo require the rwt(RICE WAVELET TOOLBOX) to work.
%
%   We set 
%
%    f_1(x)=||x||_{TV}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||z||_TV
%
%
%    f_2(x)=||H(x)||_{1}
%     We define the prox of f_1 as: 
%
%        prox_{f2,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma  ||H(z)||_1
%
%
%
%   Results
%   -------
%
%   Figure 1: Original image
%
%      This figure shows the original cameraman image. 
%
%   Figure 2: Depleted image
%
%      This figure shows the image after the addition of noise.
%
%   Figure 3: Reconstruted image
%
%      This figure shows the denoised image thanks to the algorithm.
%
%   References:
%     P. Combettes and J. Pesquet. Proximal splitting methods in signal
%     processing. Fixed-Point Algorithms for Inverse Problems in Science and
%     Engineering, pages 185-212, 2011.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_prox_multi_functions.php

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

 
% Author: Nathanael Perraudin, Gilles Puy
% Date: sept 30 2011
%

% Initialisation

clear all;
close all;
clc;



% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();


% setting different parameter for the simulation

tau1 = 2e1; %parameter for the problem
tau2 = 3e0; %parameter for the problem

% Original image
im_original=double(imread('misc/cameraman.tif'));

% Displaying original image
figure(1)
imagesc(im_original);
colormap gray;
hold on;
title('Original image');
axis off          
axis image        


% Depleted image
sigma=10;
b=im_original+sigma^2*rand(size(im_original));


% Displaying depleted image
figure(2)
imagesc(b);
colormap gray;
hold on;
title('Depleted image');
axis off          
axis image        




% setting the function f1 

% for the TV norm
param2.verbose=1;
param2.max_iter=100;
param2.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

g1.prox=@(x, T) prox_tv(x, T*tau1, param2);
g1.eval=@(x) tau1*tv_norm(x);   


% for the decomposition in wavelet
L=8;
h = daubcqf(2);

A2 = @(x) mdwt(x,h,L);
A2t = @(x) midwt(x,h,L);

param3.verbose=1;
param3.max_iter=10;
param3.Psi=A2t;
param3.Psit=A2;
param3.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

if (param3.useGPU)
    param3.h = h;
    param3.L = L;
    g2.prox=@(x, T) prox_l1_wavelet_gpu(x, T*tau2, param3);
else
    g2.prox=@(x, T) prox_l1(x, T*tau2, param3);
end
g2.eval=@(x) tau2*norm(x(:),1);  


% Parameter for the prox
G={g2,g1};
param4.G=G;
param4.max_iter=10;


% solving the problem
sol=prox_sumg(b,1,param4);


% displaying the result
figure(3)
imagesc(sol);
colormap gray;
hold on;
title('Reconstructed image');
axis off         
axis image        

close_unlocbox();

