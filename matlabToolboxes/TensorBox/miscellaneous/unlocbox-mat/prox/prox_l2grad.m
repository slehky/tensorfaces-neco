function [sol,infos] = prox_l2grad(x, gamma, param)
%PROX_L2grad Proximal operator of the 2 norm of the gradient in 1 dimension
%   Usage:  sol=prox_l2grad(x, gamma)
%           sol=prox_l2grad(x, gamma, param)
%           [sol, infos]=prox_l2grad(x, gamma, param)
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
%   arrays, the function operates on the first 
%   dimension.
%
%   PROX_L2GRAD(x, gamma, param) solves:
%
%      sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||grad(z)||_2^2
%
%
%   param is a Matlab structure containing the following fields:
%
%    param.abasis : to use another basis than the DFT (default: 0). To be
%                     done -- Not working yet
%
%    param.weights : weights if you use a an array.
%
%    param.verbose : 0 no log, 1 a summary at convergence, 2 print main
%     steps (default: 1)
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
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_l2grad.php

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
if ~isfield(param, 'weights'), param.weights = 1; end
if ~isfield(param, 'deriveorder'), param.deriveorder = 1; end
if ~isfield(param, 'abasis'), param.abasis = 0; end
if ~isfield(param, 'tight'), param.tight = 1; end
if ~isfield(param, 'nu'), param.nu = 1; end
if ~isfield(param, 'tol'), param.tol = 1e-3; end
if ~isfield(param, 'maxit'), param.maxit = 200; end
if ~isfield(param, 'At'), param.At = @(x) x; end
if ~isfield(param, 'A'), param.A = @(x) x; end

warning=0;
% test the parameters
test_gamma(gamma,warning);
test_weights(param.weights,warning);

if param.abasis
    error('This option is not currently supported, please contact the developer')
end

p=param.deriveorder;

if param.tight
    % useful function
    h=@(t) 1./(1+param.weights*param.nu*gamma*t).^p;

    % size of the signal

    temp=param.A(x);

    if (size(temp,1)==1) || (size(temp,2)==1)
        L=size(x,1);
        l=(0:L-1)';
        lambda=2-2*cos(2*pi*l/L);        

        %filtering
        sol=x+1/param.nu*param.At(ifft(fft(temp).*h(lambda))-temp);

    elseif ismatrix(x)
        L=size(x,1);
        Q=size(x,2);
        l=(0:L-1)';
        q=(0:Q-1);
        lambda=(2-2*cos(2*pi*l/L))*(2-2*cos(2*pi*q/Q));


        sol=x+1/param.nu*param.At(ifft2(fft2(temp).*h(lambda))-temp);
    else
       error('Dimension error: signal with dimention higher than 2 are not supported yet');
    end



    % one iteration
    iter=1;

    curr_norm=norm(gradient(sol),'fro')^2;
    crit='--';
else % non tight frame case (gradient descent)
    
    % Initialization
        % FISTA algorithm
        x_n=f1.prox(u_n-param.gamma*f2.grad(u_n), param.gamma);
        tn1=(1+sqrt(1+4*tn^2))/2;
        u_n=x_n+(tn-1)/tn1*(x_n-sol);
        %updates
        sol=x_n;
        tn=tn1;
end

% Summary
if param.verbose>=1
   fprintf('  Prox_l2grad: %i iteration(s), ||grad(x)||^2=%g\n',iter,curr_norm);
end


infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=curr_norm;
infos.crit=crit;
infos.time=toc(t1);

end




