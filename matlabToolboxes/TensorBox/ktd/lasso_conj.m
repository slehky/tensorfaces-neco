function [sol, infos,objectiv] = sparse_c(Y,options)
% Lasso with conjugate condition.
%
% Sparse Approximation with conjugate condition
%
% This algorithm approximates a matrix Y by a (nonnegative) matrix X of
% rank-R by solving the optimization problem
%
%    min 1/2 || Y - X ||_F^2 + gamma * || X ||_1    (1)
%    subject to F'*vec(X) = 0
%
% where || X ||_* denotes the nuclear norm of the matrix X.
%
%
% The second problem is solved using the proximal splitting method
%
%     min f1(X) + f2(x)  
%
% with
%     f1(X) = 1/2 || Y - X ||_F^2  s.t F'*vec(X) = 0   
%           prox_{lda*f1}(Z) = (I - F*F') * (lda*Y + Z)/(1+2*lda)
%     f2(X) = gamma * ||X||_1
%           prox_{lda * f2}(Z)= Soft(Z,lda)
%
%
% Phan Anh-Huy, September, 2015.
%
if ~exist('options','var'), options = struct; end
param = parseInput(options);
if nargin == 0
    sol = param; return
end

SzY = size(Y); 
 
F =[];
if isfield(param,'F') && ~isempty(param.F)
    F = param.F;
end
if isfield(param,'Yref') && ~isempty(param.Yref)
    F = param.Yref;
end
if isempty(F);
    F = 0;  
else
    [F,foe] = qr(F,0);
end

param.maxit = param.maxiters;

y= Y(:);

tau=param.tau;

Ix = param.kron_unfolding_size;
if ~isempty(Ix)
    Ia = SzY./Ix;
end


% sigma_noise = param.epsilon;
%noise_level = min(normY*.95,max(normY*.90,sqrt(nnz(Y(:)))*sigma_noise)); % noise power which should be gradually reduced.
% noise_level = sqrt(nnz(Y(:)))*sigma_noise; % noise power which should be gradually reduced.
noise_level = param.epsilon;

% Y = options.A(Y);% forward transform
% if F~=0
%     F = options.A(reshape(F,SzY));% forward transform
%     F = F(:);
% end

%% Define the prox of f2 see the function proj_B2 for more help

% setting the function f1  =  |y-x|^2 subject F'*x = 0;
if F~=0
    % Full expression
    %f2.prox=@(x,gamma) reshape((x(:)+gamma*y - F*(F'*(x(:)+gamma*y)))/(1+gamma),SzY);
    
    % Fast implementation
    yF = y - F*(F'*y);
    if isempty(noise_level) || isinf(noise_level) % if noise variance is not used to bound the approximation error
        f2.prox=@(x,gamma) reshape((x(:)- F*(F'*x(:)) - yF)/(1+gamma) + yF,SzY);
    else % if noise variance is used to bound the approximation error
        f2.prox=@(x,gamma) reshape(prox_l2_conj(x,F,yF,noise_level),SzY);
    end
    
else
    if isempty(noise_level) ||  isinf(noise_level) % if noise variance is not used to bound the approximation error
        f2.prox=@(x,gamma) reshape((x(:)-y)/(1+gamma)+y,SzY);
    else % if noise variance is used to bound the approximation error
        f2.prox=@(x,gamma) reshape(prox_l2_bound(x,y,noise_level),SzY);
    end
    
end
f2.eval=@(x) norm(x(:)-y(:))^2;

% for the sparsity constraint l1-norm
param1.verbose=1;
param1.maxit=100;
param1.useGPU = 0;   % Use GPU for the TV prox operator.
param1.A = param.A; % : Forward operator (default: Id).  wavelet or linear operator
param1.At = param.At;% Adjoint operator (default: Id).

if isempty(Ix)
    if param.autothresh == true
        f1.prox=@(x, gamma) prox_l1(x,wthresh_blk2(tau*x), param1);
    else
        f1.prox=@(x, gamma) prox_l1(x,gamma*tau, param1);
    end
    
%     f1.prox=@(x, gamma) prox_l1(x,thselect(tau*x(:),'rigrsure'), param1); % with threshold selected from the signals 
% thr = wbmpen(x,l,sigma,alpha)

    %f1.prox=@(x, gamma) prox_l1(wthresh_blk(tau*x),0, param1);
    

%     f1.prox=@(x, gamma) prox_l1(x,sqrt(2*log(numel(x))) * median(abs(tau*x(:))), param1); % with threshold selected from the signals 
    
    f1.eval=@(x) tau*norm(reshape(param1.A(x),[],1),1);   
  
else
    f1.prox=@(x, gamma) kron_folding(prox_l1(kron_unfolding(x,Ix),gamma*tau, param1),Ix,Ia);
%     f1.prox=@(x, gamma) kron_folding(prox_l1(kron_unfolding(x,Ix),thselect(tau*x(:),'rigrsure'), param1),Ix,Ia);
    f1.eval=@(x) tau*norm(reshape(param1.A(kron_unfolding(x,Ix)),[],1),1);   
end

% setting the frame L
L= @(x) x;

% solving the problem
if isempty(param.init)
    [sol, infos,objectiv]=admm(Y,f1,f2,L,param);
else
    [sol, infos,objectiv]=admm(param.init,f1,f2,L,param);
end


end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init',[],@(x) (isnumeric(x) || '' ||...
    ismember(x(1:4),{'zero' 'data'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('tau',1);
param.addOptional('gamma',1);
param.addOptional('A',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('At',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('F',[]); % linear operator in F'*vec(X) = 0
param.addOptional('Yref',[]); % linear operator in F'*vec(X) = 0
param.addOptional('kron_unfolding_size',[]); % linear operator in F'*vec(X) = 0
param.addOptional('size',[]);
param.addOptional('autothresh',true);
param.addOptional('abs_tol',1);
param.addOptional('epsilon',inf); % noise variance

param.parse(opts);
param = param.Results;
end


function sol = prox_l2_conj(x,F,yF,epsilon)
xF = x(:) - F*(F'*x(:));
d = xF - yF;
lambda = 1-min(epsilon/norm(d),1);
sol = d/(1+lambda) + yF;
end
 
function sol = prox_l2_bound(x,y,epsilon)
d = x(:) - y(:);
lambda = 1-min(epsilon/norm(d),1);
sol = d/(1+lambda) + y;
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



