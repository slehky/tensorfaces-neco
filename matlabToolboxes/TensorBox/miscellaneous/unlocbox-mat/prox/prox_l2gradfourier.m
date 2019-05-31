function [sol,infos] = prox_l2gradfourier(x, gamma, param)
%PROX_L2gradfourier Proximal operator of the 2 norm of the gradient in the Fourier domain
%   Usage:  sol=prox_l2gradfourier(x, gamma)
%           sol=prox_l2gradfourier(x, gamma, param)
%           [sol, infos]=prox_l2gradfourier(x, gamma, param)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   This function compute the 1 dimensional proximal operator of x. For
%   matrices, the function is applied to each column. For N-D
%   arrays, the function operates on the first dimension. 
%
%   Warning: the signal should not be centered. Indice 1 for abscissa 0.
%
%   PROX_L2GRADFOURIER(x, gamma, param) solves:
%
%      sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||grad(Fz)||_2^2
%
%
%   param is a Matlab structure containing the following fields:
%
%    param.weights : weights if you use a an array.
%
%    param.verbose : 0 no log, 1 a summary at convergence, 2 print main
%     steps (default: 1)
%
%    param.deriveorder : Order ot the derivative default 1
%
%
%   infos is a Matlab structure containing the following fields:
%
%    infos.algo : Algorithm used
%
%    param.iter : Number of iteration
%
%    param.time : Time of exectution of the function in sec.
%
%    param.final_eval : Final evaluation of the function
%
%    param.crit : Stopping critterion used 
%
%
%   See also:  proj_b1 prox_l1inf prox_l12 prox_tv
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_l2gradfourier.php

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
% Date: Jan 2013
%

% Start the time counter
t1 = tic;

% Optional input arguments
if nargin<3, param=struct; end

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'weights'), param.weights = 1; end
if ~isfield(param, 'deriveorder'), param.deriveorder = 1; end


warning=0;
% test the parameters
test_gamma(gamma,warning);
test_weights(param.weights,warning);


p=param.deriveorder;

% useful function
h=@(t) 1./(1+param.weights(:)*gamma*t').^p;

% size of the signal

L=size(x,1);
l=(0:L-1)';
lambda=2-2*cos(2*pi*l/L);



sol=x.*h(lambda)';



curr_norm=norm(gradient(1/sqrt(L)*fft(x)))^2;

% Summary
if param.verbose>=1
   fprintf('  Prox_l2gradfourier: ||grad(Fx)||^2=%g\n',curr_norm);
end

% zero iteration
iter=0;
crit='--';
infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=curr_norm;
infos.crit=crit;
infos.time=toc(t1);

end




