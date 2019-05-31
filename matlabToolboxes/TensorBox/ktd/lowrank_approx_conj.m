function [sol, infos,objectiv] = lowrank_approx_conj(Y,options)
% Low rank Approximation with conjugate condition
%
% This algorithm approximates a matrix Y by a nonnegative matrix X of
% rank-R by solving the optimization problem
%
%    min 1/2 || Y - X ||_F^2 + gamma * || X ||_*    (1)
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
%     f2(X) = gamma * ||X||_*
%           prox_{lda * f2}(Z)= SVT(Z,lda)
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
% mxY = norm(Y(:))/sqrt(numel(Y));
% Y = Y/mxY;
 
if isfield(param,'F')
    F = param.F;
else
    F = [];
end

if isempty(F) && isfield(param,'Yref')
    F = param.Yref;
end

if isempty(F);
    F = 0; % it should be an identity matrix, but set to 1.
else
    [F,foe] = qr(F,0);
end

y= Y(:);

tau=param.tau;

Ix = param.kron_unfolding_size;
if ~isempty(Ix)
    Ia = SzY./Ix;
end

param.maxit = param.maxiters;

%% Define the prox of f2 see the function proj_B2 for more help

% setting the function f1  =  |y-x|^2 subject F'*x = 0;
if F ~= 0
    Fy = y  - F* (F'*y);
    if param.nonnegativity
        %f2.prox=@(x,gamma) max(eps,reshape((x(:)+gamma*y - F*(F'*(x(:)+gamma*y)))/(1+gamma),SzY));
        f2.prox=@(x,gamma) max(eps,reshape((x(:)+gamma*Fy - F*(F'*x(:)))/(1+gamma),SzY));
    else
        %f2.prox=@(x,gamma) reshape((x(:)+gamma*y - F*(F'*(x(:)+gamma*y)))/(1+gamma),SzY);
        f2.prox=@(x,gamma) reshape((x(:)+gamma*Fy - F*(F'*x(:)))/(1+gamma),SzY);
    end
else
    if param.nonnegativity
        f2.prox=@(x,gamma) max(eps,reshape((x(:)+gamma*y)/(1+gamma),SzY));
    else
        f2.prox=@(x,gamma) reshape((x(:)+gamma*y)/(1+gamma),SzY);
    end
end
f2.eval=@(x) norm(x(:)-y(:))^2;

% for the nuclear norm
param1.verbose=1;
param1.maxit=100;
param1.useGPU = 0;   % Use GPU for the TV prox operator.
param1.maxrank = param.maxrank;

if isempty(Ix);
    
%     f1.prox=@(x, gamma) prox_nuclearnorm(x,gamma*tau, param1);
%     f1.eval=@(x) tau*norm_nuclear(x);   

    f1.prox=@(x, gamma) call_prox_nuclearnorm(1,x,gamma*tau, param1);
    f1.eval=@(x) tau*call_prox_nuclearnorm(2,x);

else
%     f1.prox=@(x, gamma) kron_folding(prox_nuclearnorm(kron_unfolding(x,Ix),gamma*tau, param1),Ix,Ia);
%     f1.eval=@(x) tau*norm_nuclear(kron_unfolding(x,Ix));   

    f1.prox=@(x, gamma) kron_folding(call_prox_nuclearnorm(1,kron_unfolding(x,Ix),gamma*tau, param1),Ix,Ia);
    f1.eval=@(x) tau*call_prox_nuclearnorm(2,kron_unfolding(x,Ix));
end


% setting the frame L
L= @(x) x;

% solving the problem
if isempty(param.init)
    [sol, infos,objectiv]=admm(Y,f1,f2,L,param);
else
    [sol, infos,objectiv]=admm(param.init,f1,f2,L,param);
end

% sol = sol*mxY;
% infos.f1_val = infos.f1_val *mxY;
% infos.f2_val = infos.f2_val *mxY;

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
param.addOptional('A',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('F',[]); % linear operator in F'*vec(X) = 0
param.addOptional('Yref',[]); % linear operator in F'*vec(X) = 0
param.addOptional('kron_unfolding_size',[]); % linear operator in F'*vec(X) = 0
param.addOptional('size',[]);
param.addOptional('nonnegativity',false); % true
param.addOptional('abs_tol',0);
param.addOptional('maxrank',[]);

param.parse(opts);
param = param.Results;
end

 

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

% Test of gamma
gamma=test_gamma(gamma);



% Useful functions
soft = @(z, T) sign(z).*max(abs(z)-T, 0);

if isempty(param.maxrank) || strcmp(param.maxrank,'none')
    [U,Sigma,V] =  svd(x,'econ');
    % [U,Sigma,V] =  mysvd(x);
else
    [U,Sigma,V] = mysvds(x,param.maxrank);
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



% 
% function [U,S,V] = mysvds(Y,maxrank)
% SzY = size(Y);
% mz = min(SzY); Mz = max(SzY);
% if (Mz/mz>5) && (Mz >= 5e3)
%     %     if SzY(1)>SzY(2)
%     %         C = Y'*Y;
%     %         [V,S] = eig(C);
%     %         S = (sqrt(diag(S)));
%     %         U = Y*V*diag(1./S);
%     %     else
%     %         C = Y*Y';
%     %         [U,S] = eig(C);
%     %         S = (sqrt(diag(S)));
%     %         V = diag(1./S)*U'*Y;
%     %     end
%     %     S = diag(S);
%     if SzY(1)>SzY(2)
%         Cy = full(Y'*Y);my = max(Cy(:));
%         if any(Cy(:))
%             [V,S] = eigs(Cy/my,maxrank);
%             S = sqrt(diag(S));
%             U = Y*V*diag(1./S);
%             S = diag(S);
%             
%             [Uq,Ur] = qr(U,0);
%             [uu,S,vv] = svd(Ur*S);
%             U = Uq*uu;
%             V = V*vv;
%         else
%             U = zeros(SzY(1),maxrank);
%             V = zeros(SzY(2),maxrank);
%             S = zeros(maxrank);
%         end
%     else
%         Cy = full(Y*Y');my = max(Cy(:));
%         if any(Cy(:))
%             [U,S] = eigs(Cy/my,maxrank);
%             S = sqrt(diag(S));
%             V = (U'*X)'*diag(1./S)*1/sqrt(my);
%             S = diag(S);
%             
%             [Vq,Vr] = qr(V,0);
%             [vv,S,uu] = svd(Vr*S);
%             U = U*uu;
%             V = Vq*vv;
%             
%         else
%             U = zeros(SzY(1),maxrank);
%             V = zeros(SzY(2),maxrank);
%             S = zeros(maxrank);
%         end
%     end
% 
% else
%     [U,S,V] = svds(Y,maxrank);
% end
% end
