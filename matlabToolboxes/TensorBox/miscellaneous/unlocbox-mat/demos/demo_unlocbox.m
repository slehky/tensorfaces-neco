%DEMO_UNLOCBOX  Tutorial for the unlocbox
%
%   The problem
%   -----------
%   Let suppose we have a image with missing pixels that we would like to
%   reconstruct. The first thing we do is asking ourself what do we know
%   about the problem.
%
%   Figure 1: Original image
%
%      This figure shows the original image(The cameraman). 
%
%   Figure 2: Measurement image
%
%      This figure shows the image after the application of the mask. Note
%      that 70% of the pixels have been removed.
%
%   Assumptions
%   -----------
%   In this particular example, we firslty assume that we know the position of the
%   missing pixels. This can be the result of a previous process on the
%   image or a simple assumption.
%
%   Secondly, we then assume that the image is not special. Thus, it
%   composed of patchs of colors and very few degradee. 
%
%   Thirdly, we supposed that the known pixels are exact. There is no
%   noise on the measurements.
%
%   Formulation of the problem
%   --------------------------
%   At this point, the problem can be expressed in a mathematical form. We
%   will simulate the masking operation by a mask
%
%   The first assumption leads to a constraint. 
%
%              Ax = y
%
%
%   We rewrite this contraint in this form 
%
%              || Ax - y ||_2^2  <=  epsilon
%
%   
%   epsilon can be chosen equal to 0 to satisfy exactly the constraint.
%   If the measurements are not certain, we usually set epsilon to
%   sigma the standart deviation of the noise.
%
%   Using the prior assumption that the image has a small TV-norm(image
%   composed of patch of color and few degradee), we will express the
%   problem as
%
%       argmin ||x||_TV s.t ||Ax-y||_2 < epsilon
%
%  
%   where b is the degraded image, I the identity and A an operator
%   representing the mask.
%
%   Proximity operator
%   ------------------
%   The UNLocBoX is using proximal splitting techniques for solving convex
%   optimization problem. Those techniques consist in dividing the problem
%   into 2 easier problem. We will iteratively minimize the TV norm and
%   perform the projection. Each function is minimized with its proximity
%   operator.
%
%   The proximity operator of a lower semi-continous convex function f is
%   defined by:
%
%        prox_{f,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma f(x)
%
%
%   The proximity operator are minimising a function without going too far
%   from a initial point. They can be assimiled as denoising operator. They
%   are also considered as generalisation of projection.
%
%   The proximity operator of the indicative function define by:
%
%                  /   0       if   x in C   
%         i_C(x) = |
%                  \  inf      otherwise
%
%
%   is simply the projection onto the set C.
%
%   The constraint can be inserted in the objective function
%   thanks to the help of the indicative function. Then we recover the
%   general formulation used for the solver of this toolbox.
%
%   Solving the problem
%   -------------------
%
%   The UNLocBoX solvers takes as input function with their proximity
%   operator or with their gradient. In this exemple, we need to provide
%   two functions:
%
%    f_1(x)=||x||_{TV}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma ||z||_TV
%
%
%     This function is defined in Matlab using:
%
%           paramtv.verbose=1;
%           paramtv.maxit=50;
%           f1.prox=@(x, T) prox_tv(x, T, paramtv);
%           f1.eval=@(x) tv_norm(x);   
%
%     This function is a structure with two fields. First, f1.prox is an
%     operator taking as imput x and T and evaluating the proximity
%     operator of the function (T plays the role of gamma is the
%     equation above). Second, and sometime optional, f1.eval is also an
%     operator evaluting the function in x.
%
%     The proximal operator of the TV norm is already encoded in the
%     UNLocBoX by the function ?prox_tv?. We tune it by setting the maximum
%     number of iteration and a verbose level. Other parameter are also
%     availlable (See documentation).
%
%      paramtv.verbose select the display level (0 no log, 1 summary at
%       convergence and 2 display all steps).
%
%      paramtv.maxit define the maximum number of iteration.
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
%        argmin_{z} ||x - z||_2^2   s.t.  || A z - y||_2 < epsilon
%
%
%     It is simply a projection on the B2-ball. In matlab, we write:
%
%           param_proj.epsilon=0;
%           param_proj.A=A;
%           param_proj.At=A;
%           param_proj.y=y;
%           f2.prox=@(x,T) proj_b2(x,T,param_proj);
%           f2.eval=@(x) eps;
%
%     The prox field of the sturcture is in that case the operator
%     computing the projection. Since we suppose that the constraint is
%     satisfied. The value of the indicative function is thus 0. For
%     implementation reason, it is better to set the value of the operator
%     f2.eval to eps than to 0.
%
%   At this point, a solver need to be selected. The UNLocBoX contains many
%   different solvers. You can try different of them and observe different
%   convergence speed. Just remember that some solvers are optimized for
%   specific problems. In this tutorial, we present two of them
%   ?forward_backward? and ?douglas_rachford?. Both of them takes as imput
%   two functions (they have generalization taking more functions), a 
%   starting point and some optional parameters. 
%
%   Results
%   -------
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
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_unlocbox.php

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
% Date: fev 23 2012
%

%% Initialisation
clear all;
close all;
clc;

%% Loading toolbox
global GLOBAL_useGPU;
addpath ../
init_unlocbox();



%% Load an image

% Original image
im_original=double(imread('misc/cameraman.tif'));

% Displaying original image
imagescgray(im_original,1,'Original image');  


%% Creation of the problem

% Create a matrix with randomly 70 % of zeros entry
matA=rand(size(im_original));
matA=(matA>0.7);
% Define the operator
A=@(x) matA.*x;


% Depleted image
y=matA.*im_original;

% Displaying depleted image
imagescgray(y,2,'Depleted image');  


%% Setting the proximity operator


% setting the function f1 (norm TV)
paramtv.useGPU = GLOBAL_useGPU;   % Use GPU for the TV prox operator.
paramtv.verbose=1;
paramtv.maxit=50;
f1.prox=@(x, T) prox_tv(x, T, paramtv);
f1.eval=@(x) tv_norm(x);   

% setting the function f2 
param_proj.epsilon=0;
param_proj.A=A;
param_proj.At=A;
param_proj.y=y;
f2.prox=@(x,T) proj_b2(x,T,param_proj);
f2.eval=@(x) eps;


%% Solver

% setting different parameters for the simulation
param.verbose=1;    % display parameter
param.maxit=200;    % maximum number of iterations
param.tol=10e-5;    % tolerance to stop iterating
param.gamma=10;     % Convergence parameter
% solving the problem with ppxa
sol = ppxa(y,{f1,f2},param);
% solving the problem with forward backard
% sol = douglas_rachford(y,f1,f2,param);

%% Displaying the result
imagescgray(sol,3,'Reconstructed image');   

%% Close the UNLcoBoX
close_unlocbox();

