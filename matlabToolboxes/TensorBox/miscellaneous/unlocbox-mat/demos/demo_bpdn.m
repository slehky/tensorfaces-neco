%DEMO_BPDN Deconvolution demonstration (Debluring)
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
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_bpdn.php

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
% Date: 14 May 2013
%

%% Initialisation

    clear all;
    close all;
    clc;


% Loading toolbox
    addpath ../
    init_unlocbox();



%% Creation of the problem
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






% Creation of the blur
    % Gaussian round blur
    sigma=0.1;
    [x, y] = meshgrid(linspace(-1, 1, length(im_original)));
    r = x.^2 + y.^2;
    G = exp(-r/(2*sigma^2));

    % Blur operators
    A=@(x) real(ifft2(fftshift(G).*(fft2(x))));
    At=@(x) real(ifft2(conj(fftshift(G)).*(fft2(x))));
    
% Depleted image
    b=A(im_original);



% Displaying depleted image
    figure(2)
    imagesc(b);
    colormap gray;
    hold on;
    title('Depleted image');
    axis off          
    axis image


% Prior assumption about the image (sparse in Wavelet)
    L=8;
    h = daubcqf(2);
    % Wavelet operator
    Psi = @(x) mdwt(x,h,L);
    Psit = @(x) midwt(x,h,L);



% setting different parameter for the simulation
    % General parameter
    param.verbose=1;    % display parameter
    param.maxit=50;     % maximum iteration
    param.tol=10e-6;    % tolerance to stop iterating

    % Paramter for the meaurements
    epsilon=0;          % fidelity (radius of the B2 ball)
    param.nu_b2=1;      % Lipshitz constant of A
    param.tight_b2=0;   % If A is tight
    param.maxit_b2=100; % Maximum number of iteration for the projection 
                        % onto the ball b2.
    param.tol_b2=1e-5;  % Tolerance to stop iterating
                        

    % Parameter for the L1 prior assumption
    param.maxit_l1=50;  % Maximum number of itertaion for the minimization 
                        % of the L1 norm.
    param.tight_l1=1;   % If Psi is tight
    param.nu_l1=1;      % See documentation of solve_bpdn 



% solving the problem
    sol=solve_bpdn(b,epsilon, A, At, Psi, Psit, param);

% displaying the result
    figure(3)
    imagesc(abs(sol));
    colormap gray;
    hold on;
    title('Reconstructed image');
    axis off         
axis image        

close_unlocbox();

