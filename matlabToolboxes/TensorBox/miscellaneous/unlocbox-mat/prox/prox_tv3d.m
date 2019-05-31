function [sol,infos] = prox_tv3d(b, gamma, param)
%PROX_TV3D Total variation proximal operator
%   Usage:  sol=prox_tv3d(x, gamma)
%           sol=prox_tv3d(x, gamma,param)
%           [sol, infos]=prox_tv3d(...)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   PROX_TV3D(y, gamma, param) solves:
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
%   See also:  prox_l1 prox_tv
%
%   References:
%     A. Beck and M. Teboulle. Fast gradient-based algorithms for constrained
%     total variation image denoising and deblurring problems. Image
%     Processing, IEEE Transactions on, 18(11):2419-2434, 2009.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_tv3d.php

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


% Author: Gilles Puy, Nathanael Perraudin, William Guicquero
% Date: October 15, 2010
%

% Start the time counter
t1 = tic;

% Optional input arguments
if nargin<3, param=struct; end

if ~isfield(param, 'tol'), param.tol = 1e-4; end
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'maxit'), param.maxit = 200; end

% Test of gamma
gamma=test_gamma(gamma);

% Initializations
%[r, s, k] = gradient_op3d(b*0);
[r, s, k] = gradient_op3d(b*0,1,1,1);
pold = r; qold = s; kold = k;
told = 1; prev_obj = 0;

% Main iterations
if param.verbose > 1
    fprintf('  Proximal TV operator:\n');
end
for iter = 1:param.maxit
    
    % Current solution
    %sol = b - gamma*div_op3d(r, s, k);
    sol = b - gamma*div_op3d(r, s, k, 1, 1, 1);
    
    % Objective function value
    obj = .5*norm(b(:)-sol(:), 2)^2 + gamma * TV_normnD(sol,[1; 1; 1]);
    %obj = .5*norm(b(:)-sol(:), 2)^2 + gamma * TV_norm3d(sol);
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
    %[dx, dy, dz] = gradient_op3D(sol);
    [dx, dy, dz] = gradient_op3d(sol,1,1,1);
    r = r - 1/(8*gamma) * dx;
    s = s - 1/(8*gamma) * dy;
    k = k - 1/(8*gamma) * dz;
    weights = max(1, sqrt(abs(r).^2+abs(s).^2+abs(k).^2));
    %weights = max(1, abs(r)+abs(s)+abs(k));
    p = r./weights;
    q = s./weights;
    o = k./weights;
    
    
    % FISTA update
    t = (1+sqrt(4*told^2))/2;
    r = p + (told-1)/t * (p - pold); pold = p;
    s = q + (told-1)/t * (q - qold); qold = q;
    k = o + (told-1)/t * (o - kold); kold = o;
    told = t;
    
end

% Log after the minimization
if ~exist('crit_TV', 'var'), crit = 'MAX_IT'; end
if param.verbose >= 1
    fprintf(['  Prox_TV: obj = %e, rel_obj = %e,' ...
        ' %s, iter = %i\n'], obj, rel_obj, crit, iter);
end

infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=obj;
infos.crit=crit;
infos.time=toc(t1);

end

