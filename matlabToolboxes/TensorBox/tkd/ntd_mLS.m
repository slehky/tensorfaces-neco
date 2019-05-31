function [T,A,G,fit,iter] = ntd_mLS(Y,R,opts)
% Alpha NTD algorithm using alpha divergence
% INPUT
% Y     :   tensor with size of I1 x I2 x ... x IN
% R     :   size of core tensor R1 x R2 x ... x RN: [R1, R2, ..., RN]
% opts  :   struct of optional parameters for algorithm (see defoptions)
%   .tol:   tolerance of stopping criteria (explained variation)     (1e-6)
%   .maxiters: maximum number of iteration                             (50)
%   .init:  initialization type: 'random', 'eigs', 'nvecs' (HOSVD) (random)
%   .alpha: alpha parameter                                             (1)
%   .nonlinearproj: apply half-wave rectifying or not                   (1)
%   .orthoforce:  orthogonal constraint to initialization using ALS
%   .updateYhat:  update Yhat or not, using in Fast Alpha NTD with ell-1
%                 normalization for factors                             (0)
%   .ellnorm:   normalization type                                      (1)
%   .projector: projection type for reconstructed tensor Yhat: max,(real)
%   .fixsign:  fix sign for components of factors.                      (1)
%
% Copyright by Anh Huy Phan, Andrzej Cichocki
% Ver 1.0 12/2008, Anh Huy Phan
N = ndims(Y);
% Set algorithm parameters from input or by using defaults
defoptions = struct('tol',1e-6,'maxiters',50,'init','random','ldasparse',0,...
    'nonlinearproj',1,'orthoforce',1,'updateYhat',0,'ellnorm',1,...
    'projector','real','fixsign',1,'ldaw',0,'ldab',0,'testQ',98,'verbose',1,...
    'lda_ortho',zeros(N,1));
if ~exist('opts','var')
    opts = struct;
end
opts = scanparam(defoptions,opts);

% Extract number of dimensions and norm of Y.

In = size(Y);
if numel(R) == 1
    R = R(ones(1,N));
end
if numel(R) < N
    R(N) = In(N);
end

if numel(opts.lda_ortho) == 1
    opts.lda_ortho = opts.lda_ortho(ones(1,N));
end

if numel(opts.ldasparse) == 1
    opts.ldasparse = opts.ldasparse(ones(1,N));
end

%% Set up and error checking on initial guess for U.
[A,G] = ntd_initialize(Y,opts.init,opts.orthoforce,R);
G = max(eps,double(G));
G = tensor(G);

%%
normY = norm(Y);
fprintf('\nmultiplicative LS NTD:\n');
AtA = cellfun(@(x) x'*x ,A,'uni',0);
fitold = inf;fitarr =[];
%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(.0001) % force to interrupt
    % Iterate over all N modes of the tensor
    
    for n = 1:N
        YtA = ttm(Y,A,-n,'t');
        num = double(ttt(tensor(YtA),G,setdiff(1:N,n)));
        GtA = ttm(G,AtA,-n,'t');
        den = ttt(GtA,G,setdiff(1:N,n));
        den = A{n}*den.data +...
            + opts.ldasparse(n) +...
            + opts.lda_ortho(n) * A{n} * (ones(R(n))- eye(R(n))) +eps;
        A{n} = A{n}.* num./den;
        A{n} = max(eps,A{n});
        
        if opts.ellnorm>0
            ellA = (sum(A{n}.^opts.ellnorm)).^(1/opts.ellnorm)+eps;
            A{n} = bsxfun(@rdivide,A{n},ellA);
%             G = ttm(G,diag(ellA),n);
        end
        AtA{n} = A{n}'*A{n};
    end
    
    
    num = tensor(ttm(YtA,A,N,'t'));%
    den = ttm(GtA,AtA,N,'t') + eps;
%     num = tensor(ttm(Y,A,'t'));
%     den = ttm(G,AtA) + eps;
    G = G.*(num.data./den.data);
    G = tensor(max(G.data,eps));
    Yhat = ttensor(G,A);
    if (mod(iter,5) ==1) || (iter == opts.maxiters)
        % Compute fit
        normresidual = sqrt(normY^2 + norm(Yhat)^2 -2*innerprod(Y,Yhat));
        fit = 1 - (normresidual/normY);        %fraction explained by model
        fitchange = abs(fitold - fit);
        fprintf('Iter %2d: fit = %e fitdelta = %7.1e\n', ...
            iter, fit, fitchange);                  % Check for convergence
        if (fitchange < opts.tol) && (fit>0)
            break;
        end
        fitarr = [fitarr fit];
        fitold = fit;
    end
    
    
end
%% Compute the final result
T = ttensor(G,A);
fit = fitarr;
end
