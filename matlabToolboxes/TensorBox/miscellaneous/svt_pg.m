function [X,output] = svt_pg(Y, lambda, gamma, maxiters, initial,tol)
% Proximal gradient algorithm for Singular Value thresholding
%
% The algorithm minimizes the nuclear norm of X
%     min \|X\|_*
%     subject to Omega(X) = Omega(Y)
%
% where Omega(X) takes only observed data from X and ignores the rest part
% which can be considered missing.
%
% This is equipvalent to the following problem which can be solved using
% the proximal gradient method 
%     min \gamma \|X\|_* + 1/2 * \|y - Omega(X)\|^2
% 
%  where y = Omega(Y) comprises only observed entries of Y.
%
if nargin < 6
    tol = 1e-8;
end
if nargin < 5
    initial = [];
end

Omega = isnan(Y);Omegac = ~Omega;
y = Y(Omegac); % Observed data
normy = norm(y(:));
X = Y;
if isempty(initial)
    X(Omega) = 0;
else
    X(Omega) = initial(Omega);
end
gammac = gamma*lambda;

relerror = zeros(maxiters, 1); cost = zeros(maxiters, 1);rankX = zeros(maxiters, 1);
for k = 1:maxiters
    % Proximal gradient method
%     Xomega_old = X(Omega); 
    
    X(Omegac) = (1-lambda) * X(Omegac) + lambda * y;
    [X,S] = prox_nuclearnorm(X,gammac);
    
    % Relative error and cost value
    error = norm(y - X(Omegac));
    cost(k) = gamma*sum(S(:)) + 1/2*error^2;
    relerror(k) = error/normy;
    rankX(k) = sum(S>0);
    
    if (k>10) && (abs(relerror(k)-relerror(k-1))<=tol) && ...
            (abs(cost(k)-cost(k-1))< 1e-3)
        break
    end
end
relerror(k+1:end) = [];cost(k+1:end) = [];rankX(k+1:end) = [];
X(Omegac) = y;
output = struct('cost',cost,'relerror',relerror,'rank',rankX);
end

function [x,Sshr] = prox_nuclearnorm(v, lambda)
% [U,S,V] = svd(v,'econ');S = diag(S);
[U,S,V] = msvd(v);S = diag(S);
Sshr = max(0, S - lambda);
is = Sshr>0;
x = U(:,is)*diag(Sshr(is))*V(:,is)';
end
