function [sol,infos,objectiv] = solve_bpdn(y, epsilon, A, param)
%SOLVE_BPDN Solve BPDN (basis pursuit denoising) problem
%   Usage: sol = solve_bpdn(y, epsilon, A, At, Psi, Psit, param)
%          sol = solve_bpdn(y, epsilon, A, At, Psi, Psit)
%          [sol,infos,objectiv] = solve_bpdn(...)
%
%   Input parameters:
%         y     : Measurements
%         epsilon: Radius of the L2 ball
%         A     : Operator
%         At    : Adjoint of A
%         Psi   : Operator
%         Psit  : Adjoint of Psi
%         param : Optional parameter
%   Output parameters:
%         sol   : Solution
%         infos : Structure summarizing informations at convergence
%         objectiv: vector (evaluation of the objectiv function each iteration)
%
%   sol = solve_BPDN(y, A, At, Psi, Psit, param) solves:
%
%      sol arg min ||Psi x||_1   s.t.  ||y-A x||_2 < epsilon
%
%
%   Y contains the measurements. A is the forward measurement operator and
%   At the associated adjoint operator. Psit is a sparfying transform and Psi
%   its adjoint. PARAM a Matlab structure containing the following fields:
%
%   General parameters:
% 
%    param.verbose : 0 no log, 1 print main steps, 2 print all steps.
%
%    param.maxit : max. nb. of iterations (default: 200).
%
%    param.tol : is stop criterion for the loop. The algorithm stops if
%
%         (  n(t) - n(t-1) )  / n(t) < tol,
%      
%
%     where  n(t) = Psi(x)|| is the objective function at iteration t*
%     by default, tol=10e-4.
%
%    param.gamma : control the converge speed (default: 1e-1).
% 
% 
%   Projection onto the L2-ball :
%
%    param.tight_b2 : 1 if A is a tight frame or 0 if not (default = 1)
% 
%    nu_b2 : bound on the norm of the operator A, i.e.
%
%        ` ||A x||^2 <= nu * ||x||^2 
%
%
%    tol_b2 : tolerance for the projection onto the L2 ball (default: 1e-3):
%
%      epsilon/(1-tol) <= ||y - A z||_2 <= epsilon/(1+tol)
%
%    
%    maxit_b2 : max. nb. of iterations for the projection onto the L2
%     ball (default 200).
% 
% 
%   Proximal L1 operator:
%
%    tol_l1 : Used as stopping criterion for the proximal L1
%     operator. Min. relative change of the objective value between two
%     successive estimates.
%
%    maxit_l1 : Used as stopping criterion for the proximal L1
%     operator. Maximum number of iterations.
% 
%    param.nu_l1 : bound on the norm^2 of the operator Psi, i.e.
%
%        ` ||Psi x||^2 <= nu * ||x||^2 
%
% 
%    param.tight_l1 : 1 if Psit is a tight frame or 0 if not (default = 1)
% 
%    param.weights : weights (default = 1) for a weighted L1-norm defined
%     as:
%
%        sum_i{weights_i.*abs(x_i)}
%
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
%    param.final_eval : Final evaluation of the objectivs functions
%
%    param.crit : Stopping critterion used 
%
%    param.rel_norm : Relative norm at convergence 
%
%    param.residue : Final residue 
%
%
%   The problem is solved thanks to a Douglas-Rachford splitting
%   algorithm.
%
%   Demos: demo_bpdn demo_weighted_l1
%
%   References:
%     P. Combettes and J. Pesquet. A douglas-rachford splitting approach to
%     nonsmooth convex variational signal recovery. Selected Topics in Signal
%     Processing, IEEE Journal of, 1(4):564-574, 2007.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//solver/solve_bpdn.php

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

% Author: Gilles Puy, Nathanael Perraudin
% Date: Nov. 1, 2012
%

% Start the time counter
t1 = tic;

% Optional input arguments
if nargin<4, param=struct; end

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'tol'), param.tol = 1e-4; end
if ~isfield(param, 'maxit'), param.maxit = 200; end
if ~isfield(param, 'gamma'), param.gamma = 1e-2; end
if ~isfield(param, 'pos_l1'), param.pos_l1 = 0; end

% Input arguments for projection onto the L2 ball
param_b2.A = A;  param_b2.At = A;  
param_b2.y = y; param_b2.epsilon = epsilon;
param_b2.verbose = param.verbose;
if isfield(param, 'nu_b2'), param_b2.nu = param.nu_b2; end
if isfield(param, 'tol_b2'), param_b2.tol = param.tol_b2; end
if isfield(param, 'tight_b2'), param_b2.tight = param.tight_b2; end
if isfield(param, 'maxit_b2')
    param_b2.maxit = param.maxit_b2;
end

% Input arguments for prox L1
param_l1.pos = param.pos_l1;
param_l1.verbose = param.verbose; param_l1.tol = param.tol;
if isfield(param, 'nu_l1')
    param_l1.nu = param.nu_l1;
end
if isfield(param, 'tight_l1')
    param_l1.tight = param.tight_l1;
end
if isfield(param, 'maxit_l1')
    param_l1.maxit = param.maxit_l1;
end
if isfield(param, 'tol_l1')
    param_l1.tol = param.tol_l1;
end
if isfield(param, 'weights')
    param_l1.weights = param.weights;
else
    param_l1.weights = 1;
end

% Initialization
xhat = A(y); yA = xhat;
normyy = norm(y)^2 - norm(yA)^2;
[~,~,prev_norm,iter,objectiv,~] = convergence_test();

param_b2.nu = 1;
% Main loop
while 1
    
    %
    if param.verbose >= 1
        fprintf('Iteration %i:\n', iter);
    end
    
    % Projection onto the L2-ball
    %[sol2, param_b2.u] = proj_b2(xhat, NaN, param_b2);
     
    temp = A(xhat-y);
    normd2 = sqrt(norm(temp)^2+normyy);
    alpha = min(epsilon/normd2, 1);
    sol = xhat + 1/param_b2.nu * temp * (alpha-1);
    
    
    % Global stopping criterion
    dummy = A(sol);
    curr_norm = sum(param_l1.weights(:).*abs(dummy(:)));    
    [stop,rel_norm,prev_norm,iter,objectiv,crit] = convergence_test(curr_norm,prev_norm,iter,objectiv,param);
    if stop
        break;
    end
    if param.verbose >= 1
        fprintf('  ||x||_1 = %e, rel_norm = %e\n', ...
            curr_norm, rel_norm);
    end

    
    % Proximal L1 operator
    xhat = 2*sol - xhat;
    temp = prox_l1(xhat, param.gamma, param_l1);
    xhat = temp + sol - xhat;
     
end


% Log
if param.verbose>=1
    % L1 norm
    fprintf('\n Solution found:\n');
    fprintf(' Final L1 norm: %e\n', curr_norm);
    
    % Residual
    dummy = A(sol); res = norm(y(:)-dummy(:), 2);
    fprintf(' epsilon = %e, ||y-Ax||_2=%e\n', epsilon, res);
    
    % Stopping criterion
    fprintf(' %i iterations\n', iter);
    fprintf(' Stopping criterion: %s \n\n', crit);
    
end

infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=curr_norm;
infos.crit=crit;
infos.time=toc(t1);
infos.rel_norm=rel_norm;
infos.residue=res;
 
infos.f1_val = curr_norm;
infos.f2_val = res^2;


end


%%

function [sol,infos] = prox_l1(x, gamma, param)
%PROX_L1 Proximal operator with L1 norm
%   Usage:  sol=prox_l1(x, gamma)
%           sol=prox_l1(x, gamma, param)
%           [sol, infos]=prox_l1(x, gamma, param)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   PROX_L1(x, gamma, param) solves:
%
%      sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||A z||_1
%
%
%   param is a Matlab structure containing the following fields:
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
%    param.weights : weights for a weighted L1-norm (default = 1)
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
%   References:
%     M. Fadili and J. Starck. Monotone operator splitting for optimization
%     problems in sparse recovery. In Image Processing (ICIP), 2009 16th IEEE
%     International Conference on, pages 1461-1464. IEEE, 2009.
%
%     A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding
%     algorithm for linear inverse problems. SIAM Journal on Imaging
%     Sciences, 2(1):183-202, 2009.
%
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_l1.php

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


% Author: Gilles Puy, Nathanael Perraudin
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
if ~isfield(param, 'At'), param.At = @(x) x; end
if ~isfield(param, 'A'), param.A = @(x) x; end
if ~isfield(param, 'weights'), param.weights = 1; end
if ~isfield(param, 'pos'), param.pos = 0; end

% test the parameters
gamma=test_gamma(gamma);
param.weights=test_weights(param.weights);

% Useful functions




% Projection
if param.tight && ~param.pos % TIGHT FRAME CASE
    
    temp = param.A(x);
    sol = x + 1/param.nu * param.At(soft_threshold(temp, ...
        gamma*param.nu*param.weights)-temp);
    crit = 'REL_OBJ'; iter = 1;
    dummy = param.A(sol);
    norm_l1 = sum(param.weights(:).*abs(dummy(:)));
    
else % NON TIGHT FRAME CASE OR CONSTRAINT INVOLVED
    
    % Initializations
    u_l1 = zeros(size(param.A(x)));
    sol = x - param.At(u_l1);
    prev_l1 = 0; iter = 0;
    
    % Soft-thresholding
    % Init
    if param.verbose > 1
        fprintf('  Proximal l1 operator:\n');
    end
    while 1
        
        % L1 norm of the estimate
        dummy = param.A(sol);
        norm_l1 = .5*norm(x(:) - sol(:), 2)^2 + gamma * ...
            sum(param.weights(:).*abs(dummy(:)));
        rel_l1 = abs(norm_l1-prev_l1)/norm_l1;
        
        % Log
        if param.verbose>1
            fprintf('   Iter %i, ||A x||_1 = %e, rel_l1 = %e\n', ...
                iter, norm_l1, rel_l1);
        end
        
        % Stopping criterion
        if (rel_l1 < param.tol)
            crit = 'REL_OB'; break;
        elseif iter >= param.maxit
            crit = 'MAX_IT'; break;
        end
        
        % Soft-thresholding
        res = u_l1*param.nu + param.A(sol);
        dummy = soft_threshold(res, gamma*param.nu*param.weights);
        if param.pos
            dummy = real(dummy); dummy(dummy<0) = 0;
        end
        u_l1 = 1/param.nu * (res - dummy);
        sol = x - param.At(u_l1);
        
        
        % for comprehension of Nathanael only
        % sol=x - param.At(ul1+A(sol)-soft_threshold(ul1+A(sol)))
        
        % Update
        prev_l1 = norm_l1;
        iter = iter + 1;
        
    end
end

% Log after the projection onto the L2-ball
if param.verbose >= 1
    fprintf(['  prox_L1: ||A x||_1 = %e,', ...
        ' %s, iter = %i\n'], norm_l1, crit, iter);
end


infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=norm_l1;
infos.crit=crit;
infos.time=toc(t1);
end
