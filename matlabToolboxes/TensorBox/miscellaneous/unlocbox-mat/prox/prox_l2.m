function [sol,infos] = prox_l2(x, gamma, param)
%PROX_L2 Proximal operator with L2 norm
%   Usage:  sol=prox_l2(x, gamma)
%           sol=prox_l2(x, gamma, param)
%           [sol, infos]=prox_l2(x, gamma, param)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   PROX_L2(x, gamma, param) solves:
%
%      sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||w (A z - y)||_2^2
%
%
%   where w are some weights.
%
%   param is a Matlab structure containing the following fields:
%
%    param.weights : weights for a weighted L2-norm (default = 1)
%
%    param.y : measurements (default: 0).
%
%    param.A : Forward operator (default: Id).
%
%    param.At : Adjoint operator (default: Id).
%
%    param.tight : 1 if A is a tight frame or 0 if not (default = 1)
%
%    param.nu : bound on the norm of the operator A (default: 1), i.e.
%
%        ` ||A x||^2 <= nu * ||x||^2 
%
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
%   See also:  proj_b2 prox_l1
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_l2.php

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
% Date: Nov 2012
%

% Start the time counter
t1 = tic;

% Optional input arguments
if nargin<3, param=struct; end

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'tight'), param.tight = 1; end
if ~isfield(param, 'nu'), param.nu = 1; end
if ~isfield(param, 'tol'), param.tol = 1e-3; end
if ~isfield(param, 'maxit'), param.maxit = 200; end
if ~isfield(param, 'y'), param.y = zeros(size(x)); end
if ~isfield(param, 'weights'), param.weights = ones(size(param.y)); end


if ~isfield(param, 'At')
    param.At = @(x) x; 
else
    % Check if the weight are correct with respect to the tight frame
    if (max(param.weights(:))~=min(param.weights(:))) && param.tight
        param.tight=0;
        
    end
    
end
if ~isfield(param, 'A')
    param.A = @(x) x; 
else
    % Check if the weight are correct with respect to the tight frame
    if (max(param.weights(:))~=min(param.weights(:))) && param.tight
        param.tight=0;    
    end

   
end

if max(param.weights(:))==min(param.weights(:))
    param.weights=max(param.weights(:));
end

% test the parameters
gamma=test_gamma(gamma);
param.weights=test_weights(param.weights);



% Projection
if param.tight % TIGHT FRAME CASE
    
    sol=(x+gamma*2*param.At(param.y.*abs(param.weights).^2))./(gamma*2*param.nu*param.weights.^2+1);
    
    
    % Infos for log...
    % L2 norm of the estimate
    dummy = (param.A(sol)-param.y);
    norm_l2 = .5*norm(x(:) - sol(:), 2)^2 + gamma *  norm(param.weights(:).*dummy(:))^2;
    % stopping criterion
    crit = 'REL_OB'; 
    % number of iteration
    iter=0;
else % NON TIGHT FRAME
    
    % Initializations
    u_n=x;
    sol=x;
    tn=1;
    prev_l2 = 0; iter = 0;
    % stepsize
    stepsize=1/(2*gamma*max(abs(param.weights)).^2*param.nu+1);
    
    % gradient
    grad= @(z) z-x+gamma*2.*param.At(param.weights.^2.*(param.A(z)-param.y));
    
    % Init
    if param.verbose > 1
        fprintf('  Proximal l2 operator:\n');
    end
    while 1
        
        % L2 norm of the estimate
        dummy = param.weights.*(param.A(sol)-param.y);
        norm_l2 = .5*norm(x(:) - sol(:), 2)^2 + gamma * norm(dummy(:))^2;
        rel_l2 = abs(norm_l2-prev_l2)/norm_l2;
        
        % Log
        if param.verbose>1
            fprintf('   Iter %i, ||w (A x- y)||_2^2 = %e, rel_l2 = %e\n', ...
                iter, norm_l2, rel_l2);
        end
        
        % Stopping criterion
        if (rel_l2 < param.tol)
            crit = 'REL_OB'; break;
        elseif iter >= param.maxit
            crit = 'MAX_IT'; break;
        end
        
        % FISTA algorithm
        x_n=u_n-stepsize*grad(u_n);
        tn1=(1+sqrt(1+4*tn^2))/2;
        u_n=x_n+(tn-1)/tn1*(x_n-sol);
        %updates
        sol=x_n;
        tn=tn1;
        
 
        % Update
        prev_l2 = norm_l2;
        iter = iter + 1;
        
    end
end

% Log after the projection onto the L2-ball
if param.verbose >= 1
    fprintf('  prox_L2: ||w (A x- y) ||_2^2 = %f, %s, iter = %i\n', ...
    norm_l2, crit, iter);

end



infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=norm_l2;
infos.crit=crit;
infos.time=toc(t1);

end

