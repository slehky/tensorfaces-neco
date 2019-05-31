%DEMO_SOUND_RECONSTRUCTION Sound time in painting demonstration
%
%   Here we try to recover missing sample of an sound. The problem can be expressed as this
%
%        argmin  ||Ax-b||^2 + tau*||G(x)||_1
%
%  
%   Where b is the degraded image, I the identity and A an operator representing the mask.
%
%   G is a linear operator projecting the signal in a sparse
%   representation. Here we are working with a Gabor transform. 
%
%   Warning! Note that this demo requires the LTFAT toolbox to work.
%
%   We set 
%
%    f_1(x)=||G(x)||_{1}
%     We define the prox of f_1 as: 
%
%        prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma  ||G(z)||_1
%
%
%    f_2(x)=||Ax-b||_2^2
%     We define the gradient as: 
%
%        grad_f(x) = 2 * A^*(Ax-b)
%
%
%   Results
%   -------
%
%   Figure 1: Original image
%
%      This figure shows the original histogram.
%
%   Figure 2: Depleted image
%
%      This figure shows the histogram after the loss of the sample (We loos 75% of the samples.)
%
%   Figure 3: Reconstructed image
%
%      This figure shows the histogram of the reconstructed sound thanks to the algorithm.
%   References:
%     P. Combettes and J. Pesquet. A douglas-rachford splitting approach to
%     nonsmooth convex variational signal recovery. Selected Topics in Signal
%     Processing, IEEE Journal of, 1(4):564-574, 2007.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//demos/demo_sound_reconstruction.php

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
% Date: sept 30 2011
%

%% Initialisation

clear all;
close all;
clc;

writefile=0;


% Loading toolbox
addpath ../
init_unlocbox(); % start the unlocbox
%ltfatstart(); % start the ltfat toolbox


% setting different parameter for the simulation
param.verbose=1; % display parameter
param.maxit=50; % maximum iteration
param.tol=10e-5; % tolerance to stop iterating
param.gamma=0.5; % stepsize (beta is equal to 2)
param.method='FISTA'; % desired method for solving the problem

tau = 1e-2; %parameter for the problem

% Original sound

[sound_original Fs]=gspi();
length_sig=length(sound_original); % Put a small number here if you want to proceed only a part a of the signal
sound_part=sound_original(1:length_sig);

% In oder to write the depleted sound somewhere
if writefile
    wavwrite(sound_part,Fs,'original.wav');
end

% Creating the problem
Mask=rand(size(sound_part));
Mask=(Mask>0.66);
% Depleted image
sound_depleted=Mask.*sound_part;
if writefile
    wavwrite(sound_depleted,Fs,'depleted.wav');
end


% setting the function f2 
f2.grad=@(x) 2*Mask.*(Mask.*x-sound_depleted);
f2.prox=@(x,T) x-Mask.*x+sound_depleted;
f2.eval=@(x) norm(Mask(:).*x(:)-sound_depleted(:))^2;


% setting the function f1


% select a gabor frame for a real signal with a Gaussian window
a=64; % size of the shift in time
M=256;% number of frequencies
F=frametight(frame('dgtreal','gauss',a,M));

% Get the framebounds
[GA,GB]= framebounds(F);

% Define the Frame operators
Psi = @(x) frana(F,x);
Psit = @(x) frsyn(F,x);

% tight frame constant
param2.nu = GB;

% set parameters
param2.verbose=2;
param2.max_iter=50;
param2.A=Psi;
param2.At=Psit;

% Since we choose a tight Gabor frame
param2.tight = 0;

f1.prox=@(x, T) prox_l1(x, T*tau, param2);
f1.eval=@(x) tau*norm(Psi(x),1);   



% solving the problem
sol=forward_backward(sound_depleted,f1,f2,param);
param.gamma=10;
sol2=douglas_rachford(sound_depleted,f1,f2,param);


%% Evaluate the result
SNR_in=snr(sound_part,sound_depleted)
SNR_fin=snr(sound_part,sol)
SNR_fin2=snr(sound_part,sol2)
% In order to write the restored sound somewhere
if writefile
    wavwrite(sol,Fs,'restored.wav');
end

dr=90;

figure(1);
plotframe(F,Psi(sound_part),Fs,dr);
title('Gabor transform of the original sound');

figure(2);
plotframe(F,Psi(sound_depleted),Fs,dr);
title('Gabor transform of the depleted sound');

figure(3);
plotframe(F,Psi(sol),Fs,dr);
title('Gabor transform of the reconstructed sound');

