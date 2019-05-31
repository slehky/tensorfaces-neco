%DEMO_RLR  Example of use of the RLR solver
%
%   We present an example of the RLR solver through an image reconstruction problem.
%   The problem can be expressed as this
%
%        argmin ||Ax-b||^2 + tau*||x||_TV
%
%  
%   Where b is the degraded image, I the identity and A an operator representing the mask.
%
%   We set 
%
%    f_1(x)=||x||_{TV}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||z||_TV
%
%
%   Results
%   -------
%
%   Figure 1: Original image
%
%      This figure shows the original Lena image. 
%
%   Figure 2: Depleted image
%
%      This figure shows the image after the application of the mask. Note
%      that 50% of the pixels have been removed.
%
%   Figure 3: Reconstruted image
%
%      This figure shows the reconstructed image thanks to the algorithm.
%   
%
%   References:
%     P. Combettes and J. Pesquet. Proximal splitting methods in signal
%     processing. Fixed-Point Algorithms for Inverse Problems in Science and
%     Engineering, pages 185-212, 2011.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_rlr.php

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
%

%% Initialisation

clear all;
close all;
clc;



% adding depency
% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();


% setting different parameter for the simulation
param.verbose=1; % display parameter
param.maxit=40; % maximum iteration
param.epsilon=10e-5; % tolerance to stop iterating
param.gamma=0.5; % stepsize (beta is equal to 2)
param.method='FISTA'; % desired method for solving the problem

tau = 1e1; %parameter for the problem

% Original image
im_original=double(imread('misc/lena_gray.bmp'));

% Displaying original image
figure(1);
imagesc(im_original);
colormap gray;
hold on;
title('Original image');
axis off          
axis image        

% Creating the problem
A=rand(size(im_original));
A=(A>0.5);
% Depleted image
b=A.*im_original;

% Displaying depleted image
figure(2)
imagesc(b);
colormap gray;
hold on;
title('Depleted image');
axis off          
axis image        

% Defining the adjoint operator
A_op = @(x) A.*x;
At_op = @(x) A.*x;


% setting the function f1 (norm TV)
param2.verbose=1;
param2.max_iter=50;
param2.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

f.prox=@(x, T) prox_tv(x, T*tau, param2);
f.eval=@(x) tau*tv_norm(x);   


% solving the problem
sol=rlr(b,f,A_op,At_op,param);


% displaying the result
figure(3)
imagesc(sol);
colormap gray;
hold on;
title('Reconstructed image');
axis off         
axis image        

close_unlocbox();

