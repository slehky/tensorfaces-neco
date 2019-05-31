function [sol, infos,objectiv,y_n,z_n] = admm(x_0,f1, f2, L, param)
%ADMM alternating-direction method of multipliers
%   Usage: sol = admm(x_0,f1,f2,L,param);
%          sol = admm(x_0,f1,f2,L);
%          [sol,infos,objectiv] = admm(...);
%
%   Input parameters:
%         x_0   : Starting point of the algorithm
%         f1    : First function to minimize
%         f2    : Second function to minimize
%         L     : Linear operation on x
%         param : Optional parameter
%   Output parameters:
%         sol   : Solution
%         infos : Structure summarizing informations at convergence
%         objectiv: vector (evaluation of the objectiv function each iteration)
%
%   ADMM (using alternating-direction method of multipliers) solves:
%
%      sol = argmin f1(x) + f2(y) such that y=Lx
%
%   
%   where
%   x is the variable.
%
%    x_0 is the starting point.
%
%    f1 is a structure representing a convex function. Inside the structure, there
%     have to be the prox of the function that can be called by f1.prox and 
%     the function itself that can be called by f1.eval. 
%     WARNING !!!  The prox of f1 is not the usual prox! But the solution to this problem:
%
%        prox_{f1, gamma }^L(z)=min_x  1/2 ||Lx-z||_2^2 + gamma f1(x)
%
%
%    f2 is a structure representing a convex function. Inside the structure, there
%     have to be the prox of the function that can be called by f2.prox and 
%     the function itself that can be called by f2.eval.
%     The prox of f2 is the usual prox:
%
%        prox_{f2, gamma }(z)=min_x  1/2 ||x-z||_2^2 + gamma f2(x)
%
%
%    L is a linear operator or a matrix (be careful with operator, they might not be frame)
%           
%
%    param a Matlab structure containing the following fields:
%
%     General parameters:
%
%      param.gamma : is the convergence parameter. By default, it's 1. (greater than 0)
%
%      param.tol : is stop criterion for the loop. The algorithm stops if
%
%           (||  y(t) - y(t-1) ||)  /  || y(t) || < tol,
%      
%
%       where  y(t) are the dual the objective function at iteration t*
%       by default, tol=10e-4.
%
%      param.maxit : is the maximum number of iteration. By default, it is 200.
% 
%      param.verbose : 0 no log, 1 print main steps, 2 print all steps.
%
%      param.abs_tol : If activated, this stopping critterion is the
%       objectiv function smaller than param.tol. By default .
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
%
%   See also:  sdmm, ppxa, generalized_forward_backward
%
%   Demos:  demo_admm
%
%   References:
%     P. Combettes and J. Pesquet. Proximal splitting methods in signal
%     processing. Fixed-Point Algorithms for Inverse Problems in Science and
%     Engineering, pages 185-212, 2011.
%     
%
%   Url: http://unlocbox.sourceforge.net/doc//solver/admm.php

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
% Date: 22 october 2012
%

% Start the time counter
t1 = tic;


% Optional input arguments
if nargin<4, param=struct; end

if ~isfield(param, 'tol'), param.tol=10e-4 ; end
if ~isfield(param, 'maxit'), param.maxit=200; end
if ~isfield(param, 'verbose'), param.verbose=1 ; end
if ~isfield(param, 'lambda'), param.lambda=1 ; end
if ~isfield(param, 'gamma'), param.gamma=1 ; end
if ~isfield(param, 'abs_tol'), param.abs_tol=1 ; end

if nargin<3 
    f2.prox=@(x) prox_L1(x, 1, param);
    f2.eval=@(x) norm(x(:),1);   
end

% test the evaluate function
[f1] = test_eval(f1);
[f2] = test_eval(f2);

if isa(L,'numeric')
   OpL= @(x) L*x;
else
   OpL= L;
end

% Initialization

curr_norm = f1.eval(x_0)+f2.eval(OpL(x_0));  
[~,~,prev_norm,~,~,~] = convergence_test(curr_norm);
[~,~,prev_rel_dual,iter,objectiv,~] = convergence_test(1);

y_n = x_0;
y_old=x_0;
z_n = zeros(size(x_0));


% Main loop
while 1
    
    %
    if param.verbose >= 1
        fprintf('Iteration %i:\n', iter);
    end
    

    % Algorithm
    x_n=f1.prox(y_n-z_n,param.gamma);
    s_n=OpL(x_n);
    y_n=f2.prox(s_n+z_n,param.gamma);
    reldual=norm(y_old(:)-y_n(:))/norm(y_n(:));

    
    z_n=z_n+s_n-y_n ;% updates
    sol=x_n; 
    y_old=y_n;
    
    % Global stopping criterion
    f1_val = f1.eval(sol);
    f2_val = f2.eval(OpL(sol));  
    curr_norm = f1_val + f2_val;
    [~,rel_norm,prev_norm,~,~,~] = convergence_test(curr_norm,prev_norm);
    [stop,~,prev_rel_dual,iter,objectiv,crit] = convergence_test(reldual,...
            prev_rel_dual,iter,objectiv,param);

    if stop && (iter  >1)
        break;
    end
    if param.verbose >= 1
        fprintf(' ||f|| = %e, rel_norm = %e\n Maximum relative distance of dual variable: %e\n', ...
            curr_norm, rel_norm, reldual);
    end
    
end

% Log
if param.verbose>=1
    fprintf('\n Solution found:\n');
    fprintf(' Final relative norm: %e\n', rel_norm );
    
    
    % Stopping criterion
    fprintf(' %i iterations\n', iter);
    fprintf(' Stopping criterion: %s \n\n', crit);
    
end

infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=curr_norm;
infos.f1_val = f1_val;
infos.f2_val = f2_val;
infos.crit=crit;
infos.time=toc(t1);
infos.rel_norm=rel_norm;

end

