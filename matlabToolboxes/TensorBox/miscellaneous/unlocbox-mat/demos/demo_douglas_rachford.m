%DEMO_DOUGLAS_RACHFORD  Example of use of the douglas_rachford solver
%
%   We present an example of the douglas_rachford solver through an image
%   reconstruction problem.
%   The problem can be expressed as this
%
%       argmin ||x||_TV s.t ||b-Ax||_2 < epsilon
%
%  
%   Where b is the degraded image, I the identity and A an operator representing the mask.
%
%   Note that the constraint can be inserted in the objective function
%   thanks to the help of the indicative function. Then we recover the
%   general formulation used for the solver of this toolbox.
%
%   We set 
%
%    f_1(x)=||x||_{TV}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||z||_TV
%
%
%    f_2 is the indicator function of the set S define by Ax-b||_2 < epsilon
%     We define the prox of f_2 as 
%
%        prox_{f2,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma i_S( x ),
%
%
%     with i_S(x) is zero if x is in the set S and infinity otherwise.
%     This previous problem has an identical solution as:
%
%        argmin_{z} ||x - z||_2^2   s.t.  ||b - A z||_2 < epsilon
%
%
%     It is simply a projection on the B2-ball.
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
%      that 70% of the pixels have been removed.
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
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_douglas_rachford.php

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



% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();

% setting different parameter for the simulation
param.verbose=1; % display parameter
param.max_iter=100; % maximum iteration
param.epsilopn=10e-5; % tolerance to stop iterating
param.gamma=1e1; % stepsize


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
A=(A>0.7);
% Depleted image
b=A.*im_original;

% Displaying depleted image
figure(2);
imagesc(b);
colormap gray;
hold on;
title('Depleted image');
axis off          
axis image        

% Define the prox of f2 see the function proj_B2 for more help
operatorA=@(x) A.*x;
epsilon2=0e-2;
param3.epsilon=epsilon2;
param3.A=operatorA;
param3.At=operatorA;
param3.y=b;


% setting the function f2 
f2.prox=@(x,T) proj_b2(x,T,param3);
f2.eval=@(x) norm(A(:).*x(:)-b(:))^2;


% setting the function f1 (norm TV)
param2.verbose=1;
param2.max_iter=50;
param2.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.

f1.prox=@(x, T) prox_tv(x, T, param2);
f1.eval=@(x) tv_norm(x);   


% solving the problem
sol=douglas_rachford(b,f1,f2,param);


% displaying the result
figure(3);
imagesc(sol);
colormap gray;
hold on;
title('Reconstructed image');
axis off         
axis image        

close_unlocbox();

