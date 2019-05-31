%DEMO_DECONVOLUTION Deconvolution demonstration (Debluring)
%
%   Here we try to deblur an image through a deconvolution problem. The
%   convolution operator is the blur
%   The problem can be expressed as this
%
%        argmin  ||Ax-b||^2 + tau*||H(x)||_1
%
%  
%   Where b is the degraded image, I the identity and A an operator representing the blur.
%
%   H is a linear operator projecting the signal in a sparse
%   representation. Here we worked with wavelet. 
%
%   Warning! Note that this demo require the rwt(RICE WAVELET TOOLBOX) to work.
%
%   We set 
%
%    f_1(x)=||H(x)||_{1}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma  ||H(z)||_1
%
%
%    f_2(x)=||Ax-b||_2^2
%     We define the gradient as: 
%
%        grad_f(x) = 2 A^*(Ax-b)
%
%
%   Results
%   -------
%
%   Figure 1: Original image
%
%      This figure shows the original lena image. 
%
%   Figure 2: Depleted image
%
%      This figure shows the image after the application of the blur.
%
%   Figure 3: Reconstructed image
%
%      This figure shows the reconstructed image thanks to the algorithm.
%
%   References:
%     P. Combettes and J. Pesquet. Proximal splitting methods in signal
%     processing. Fixed-Point Algorithms for Inverse Problems in Science and
%     Engineering, pages 185-212, 2011.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_deconvolution.php

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

%% Initialisation

clear all;
close all;
clc;


% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();



% setting different parameter for the simulation
param.verbose=1; % display parameter
param.maxit=300; % maximum iteration
param.tol=10e-9; % tolerance to stop iterating
param.gamma=0.5; % stepsize (beta is equal to 2)
param.method='FISTA'; % desired method for solving the problem

tau = 0.1; %parameter for the problem

% Original image
im_original=double(imread('misc/cameraman.tif'));

% Displaying original image
figure(1);

imagesc(im_original);
colormap gray;
hold on;
title('Original image');
axis off          
axis image        



% Creating the problem
sigma=0.1;
[x, y] = meshgrid(linspace(-1, 1, length(im_original)));
r = x.^2 + y.^2;
G = exp(-r/(2*sigma^2));


% Depleted image
b=abs(ifft2(fft2(im_original).*fftshift(G)));

% Displaying depleted image
figure(2)
imagesc(b);
colormap gray;
hold on;
title('Depleted image');
axis off          
axis image


% setting the function f2 
A=@(x) ifft2(fftshift(G).*(fft2(x)));
At=@(x) ifft2(conj(fftshift(G)).*(fft2(x)));



% setting the function f1

L=8;
h = daubcqf(2);

A2 = @(x) mdwt(x,h,L);
A2t = @(x) midwt(x,h,L);

param2.verbose=1;
param2.maxit=50;
param2.At=A2t;
param2.A=A2;
param2.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

if (param2.useGPU)
    param2.h = h;
    param2.L = L;
    f.prox=@(x, T) prox_l1_wavelet_gpu(x, T*tau, param2);
else
    f.prox=@(x, T) prox_l1(x, T*tau, param2);
end
f.eval=@(x) tau*norm(x(:),1);   


% solving the problem
sol=rlr(b,f,A,At,param);


% displaying the result
figure(3)
imagesc(abs(sol));
colormap gray;
hold on;
title('Reconstructed image');
axis off         
axis image        

close_unlocbox();

