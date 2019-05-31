% function [V rho iter] = trace_ratio_opt(A, B, dim, sigma, varargin)
% Optimize the trace ratio sum_l Tr(vl*'Al*vl)/sum_l Tr(vl*'Bl*vl)
% Input:
% - A, B: symetric positive definite matrices (nxn)
% - dim: dimension of projected space (V: nxk)
% - sigma: 'max' (maximize) or 'min' (minimize)
% - varargin: options
%   + maxit: max iteration (default 30)
%   + tol: tolerance until stop (default 1.e-06)
%   + use_shift_invert: use shift invert when we are close to the optimum
%                       (default 1). When we are close to the optimum,
%                       the eigenvalues are small, it may be better to
%                       shift the spectrum by the previously found
%                       max-eigenvalue and invert
%   + shift_invert_tol: tolerence of max eigenvalue of G to switch to
%                       shift-invert mode
%                       (default 1.e-2)
%   + use_adaptive_tol: a simple heuristic to adjust tolerence for eigensolver
%                       based on current max eigenvalue of G
%   + verbose:
%     0: quiet mode (default)
%     1: display intermediate results
%     2: display intermediate results and warnings
% Output:
% - V: projection matrix
% - rho (= trace(V'*A*V)/trace(V'*B*V))
% - iter: number of iterations
%
% Thanh Ngo, 2009
function [V rho iter] = trace_ratio_opt(A, B, dim, sigma, weights, varargin)

%    max   (\sum_{l}  w_l * trace( X_l^T * A_l * Xl))/(\sum_{l}  w_l * trace( X_l^T * B_l * Xl))
% 
dim  = 1;
params  = get_varargin(varargin);
tol     = get_params(params, 'tol', 1.e-06);
maxit   = get_params(params, 'maxit', 30);
verbose = get_params(params, 'verbose', 0);
use_adaptive_tol = get_params(params, 'use_adaptive_tol', 1);
use_si  = get_params(params, 'use_shift_invert', 1);
si_tol  = get_params(params, 'shift_invert_tol', 1.e-2);

sigma   = upper(sigma);

% A cell array of Al 

[m n] = size(A{1});
L = numel(A);

traceA = cellfun(@(x) trace(x),A,'uni',0);
traceB = cellfun(@(x) trace(x),B,'uni',0);

%scale = traceA/traceB;

% A dummy way to initialize rho
trc1 = 0;trc2 = 0;%scales = zeros(L,1);
for kl = 1:L
      
    diagA = diag(A{kl});
    diagB = diag(B{kl});
    trc1 = trc1 + sum(diagA(1:dim))*weights(kl);
    trc2 = trc2 + sum(diagB(1:dim))*weights(kl);
end
rho = trc1/trc2;


%%
if verbose
    fprintf('Iter 0: rho = %f\n', rho);
end

iter = 1;
eig_sigma = 'LA';
options.tol = tol;
options.isreal = 1;
options.issym = 1;
lambda_l = nan(L,1);
V = cell(L,1);
while 1
    
    % compute the pencil matrix
    for l = 1:L
        if strcmp(sigma, 'MAX')
            G = A{l} - rho*B{l};
        else
            G = rho*B{l} - A{l};
        end
        
        % compute largest eigenvects of G=A-rho*B
        if verbose <= 1, warning off; end
        
        if lambda_l(l) == 0
            continue;
        end
        if use_si && iter > 1 && lambda_l(l) < si_tol && ~isnan(lambda_l(l))
            if verbose
                fprintf('Use shift invert with shift = %f\n', lambda_l(l));
            end
            [Vl,lambda] = eigs(G, dim, lambda_l(l), options); lambda = diag(lambda);
        else
            [Vl,lambda] = eigs(G, dim, eig_sigma, options); lambda = diag(lambda);
        end
        if lambda >=0
            V{l} = Vl;
            lambda_l(l) = lambda(1);
        end
        if verbose <= 1, warning on; end
        
    end
    
    % compute new rho
    rho_old = rho;
    trc1 = 0;trc2 = 0;
    for l = 1:L
        if ~isnan(lambda_l(l))
            trc1 = trc1+weights(kl) * trace(V{l}'*A{l}*V{l}); % can be made faster, but Matlab does this well enough
            trc2 = trc2+weights(kl) * trace(V{l}'*B{l}*V{l});
        end
    end
    rho = trc1/trc2;
    
    if verbose
        fprintf('Iter %d: rho = %f\n', iter, rho*scale);
        
        if verbose > 1 && strcmp(sigma, 'MIN') && rho > rho_old
            warning('Rho is increasing');
        end
        
        if verbose > 1 && strcmp(sigma, 'MAX') && rho < rho_old
            warning('Rho is decreasing');
        end
        
    end
    
    if iter == maxit ...
            || abs(rho-rho_old) < tol ...
            || abs(rho-rho_old) < tol*rho
        %|| sum(lambda.*(lambda>0))<tol
        break;
    end
    
    % a simple heuristic to adjust tolerence for eigensolver
    % based on current max eigenvalue of G
    if use_adaptive_tol
        options.tol = max(lambda_l/10);
    end
    
    iter = iter+1;
end

% get params with default values
function p = get_params(params,name,defval)
if (isfield(params,name))
    p = getfield(params,name);
else
    p = defval;
end

% put varargin to a param struct
function args = get_varargin(vars)
args.foo = 0;
for i=1:2:size(vars,2)
    args = setfield(args,vars{i},vars{i+1});
end