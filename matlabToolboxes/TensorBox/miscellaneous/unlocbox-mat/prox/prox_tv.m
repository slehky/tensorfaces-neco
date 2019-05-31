function [sol, infos] = prox_tv(b, gamma, param)
%PROX_TV Total variation proximal operator
%   Usage:  sol=prox_tv(x, gamma)
%           sol=prox_tv(x, gamma,param)
%           [sol, infos]=prox_tv(...)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   This function compute the 2 dimentional TV proximal operator evaluated
%   in b. If b is a cube, this function will evaluate the TV proximal
%   operator on each image of the cube. For 3 dimention TV proximal
%   operator the function prox_tv3d can be used.
%
%   PROX_TV(y, gamma, param) solves:
%
%      sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||x||_TV
%
%
%   param is a Matlab structure containing the following fields:
%   
%    param.tol : is stop criterion for the loop. The algorithm stops if
%
%         (  n(t) - n(t-1) )  / n(t) < tol,
%      
%
%     where  n(t) = f(x)+ 0.5 X-Z_2^2 is the objective function at iteration t*
%     by default, tol=10e-4.
%
%    param.maxit : max. nb. of iterations (default: 200).
%
%    param.useGPU : Use GPU to compute the TV prox operator. Please prior 
%     call init_gpu and free_gpu to launch and release the GPU library (default: 0).
%
%    param.verbose : 0 no log, 1 a summary at convergence, 2 print main
%     steps (default: 1)
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
%   See also:  prox_l1 prox_tv3d
%
%
%   References:
%     A. Beck and M. Teboulle. Fast gradient-based algorithms for constrained
%     total variation image denoising and deblurring problems. Image
%     Processing, IEEE Transactions on, 18(11):2419-2434, 2009.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_tv.php

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


% Author: Nathanael Perraudin, Gilles Puy, Eyal Hirsch
% Date: Jan 2013
%

% Start the time counter
t1 = tic;

% for the GPU
global GLOBAL_useGPU; 

% Optional input arguments

if nargin<3, param=struct; end

if ~isfield(param, 'tol'), param.tol = 10e-4; end
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'maxit'), param.maxit = 200; end
if ~isfield(param, 'useGPU'), param.useGPU = GLOBAL_useGPU; end

% Test of gamma
gamma=test_gamma(gamma);

% Initializations
[r, s] = gradient_op(b*0);
pold = r; qold = s;
told = 1; prev_obj = 0;

% Main iterations
if param.verbose > 1
    fprintf('  Proximal TV operator:\n');
end

if param.useGPU
    sol = zeros(size(b));
    gpuResults=zeros(4, 1);
    gpuResults_ptr = libpointer('singlePtr', single(gpuResults));
    r_ptr = libpointer('singlePtr', single(r));
    s_ptr = libpointer('singlePtr', single(s));
    if sum(imag(b(:)))
        warning('Data converted to real values\n');       
    end
    b = real(b);
    b_ptr = libpointer('singlePtr', single(b));
    sol_ptr = libpointer('singlePtr', single(sol));
    Width  = size(sol, 1);
    Height = size(sol, 2);
    Depth  = size(sol, 3);
    obj = 0; rel_obj = 0; crit = 'test';

    binaryName = get_gpu_binary_name();
    result = calllib(binaryName,'ProxTV', sol_ptr, b_ptr, r_ptr, s_ptr, gpuResults_ptr, Width, Height, Depth, gamma, param.maxit, param.tol);
    sol = reshape(sol_ptr.Value, [size(b, 1) size(b, 2) size(b, 3)]);
    sol = double(sol);
    
    % Parse the return values from the GPU implementation.
    gpuResults = reshape(gpuResults_ptr.Value, [size(gpuResults, 1) size(gpuResults, 2)]);
    obj=gpuResults(1);
    rel_obj=gpuResults(2);
    iter=gpuResults(3);
    if rel_obj < param.tol
        crit = 'TOL_EPS';
    end
else
    for iter = 1:param.maxit

        % Current solution
        sol = b - gamma*div_op(r, s);

        % Objective function value
        obj = .5*norm(b(:)-sol(:), 2)^2 + gamma * sum(tv_norm(sol));
        rel_obj = abs(obj-prev_obj)/obj;
        prev_obj = obj;

        % Stopping criterion
        if param.verbose>1
            fprintf('   Iter %i, obj = %e, rel_obj = %e\n', ...
                iter, obj, rel_obj);
        end
        if rel_obj < param.tol
            crit = 'TOL_EPS'; break;
        end

        % Udpate divergence vectors and project
        [dx, dy] = gradient_op(sol);
        r = r - 1/(8*gamma) * dx; s = s - 1/(8*gamma) * dy;
        weights = max(1, sqrt(abs(r).^2+abs(s).^2));
        p = r./weights; q = s./weights;

        % FISTA update
        t = (1+sqrt(4*told^2))/2;
        r = p + (told-1)/t * (p - pold); pold = p;
        s = q + (told-1)/t * (q - qold); qold = q;
        told = t;

    end
end

% Log after the minimization
if ~exist('crit_TV', 'var'), crit = 'MAX_IT'; end
if param.verbose >= 1
    if param.useGPU
        fprintf(['  GPU Prox_TV: obj = %e, rel_obj = %e,' ...
            ' %s, iter = %i\n'], obj, rel_obj, crit, iter);
    else
        fprintf(['  Prox_TV: obj = %e, rel_obj = %e,' ...
            ' %s, iter = %i\n'], obj, rel_obj, crit, iter);
    end
end

infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=obj;
infos.crit=crit;
infos.time=toc(t1);

end

