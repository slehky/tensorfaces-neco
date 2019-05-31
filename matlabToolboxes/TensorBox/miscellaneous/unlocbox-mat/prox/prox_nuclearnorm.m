function [sol, infos] = prox_nuclearnorm(x, gamma, param)
%PROX_NUCLEARNORM Proximal operator with the nuclear norm
%   Usage:  sol=prox_nuclearnorm(x, gamma)
%           sol=prox_nuclearnorm(x, gamma, param)
%           [sol,infos]=prox_nuclearnorm(...)
%
%   Input parameters:
%         x     : Input signal.
%         gamma : Regularization parameter.
%         param : Structure of optional parameters.
%   Output parameters:
%         sol   : Solution.
%         infos : Structure summarizing informations at convergence
%
%   prox_NuclearNorm(x, gamma, param) solves:
%
%      sol = min_{z} 0.5*||x - z||_2^2 + gamma * ||x||_*
%
%   
%   param is a Matlab structure containing the following fields:
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
%   See also:  prox_l1 proj_b1 prox_tv
%
%   Url: http://unlocbox.sourceforge.net/doc//prox/prox_nuclearnorm.php

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
% Date: June 2012 EPFL
%

% Start the time counter
t1 = tic;

if nargin<3, param=struct; end

% Optional input arguments
if ~isfield(param, 'verbose'), param.verbose = 1; end
if ~isfield(param, 'maxrank'), param.maxrank= []; end

% Test of gamma
gamma=test_gamma(gamma);



% Useful functions
soft = @(z, T) sign(z).*max(abs(z)-T, 0);

if isempty(param.maxrank)
    [U,Sigma,V] =  svd(x,'econ');
    % [U,Sigma,V] =  mysvd(x);
else
    [U,Sigma,V] =  mysvds(x,param.maxrank);
end

% Shrink:
sigma = diag(Sigma);
sigma = soft(sigma,gamma);
r = sum(sigma > 0);
U = U(:,1:r); V = V(:,1:r); Sigma = diag(sigma(1:r)); nuclearNorm = sum(diag(Sigma));
sol = U*Sigma*V.';

if param.verbose >= 1
    fprintf('  prox nuclear norm: rank= %i, |x|_* = %e \n', r, nuclearNorm);
end

iter=0;
crit='--';
infos.algo=mfilename;
infos.iter=iter;
infos.final_eval=nuclearNorm;
infos.crit=crit;
infos.time=toc(t1);

end