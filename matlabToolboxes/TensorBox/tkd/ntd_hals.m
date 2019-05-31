function [T,output] = ntd_hals(Y,R,opts)
% HALS algorithms for Nonnegative Tucker Decomposition.
% For 3-way tensor decomposition, please use the fast version NTD_HALS3.
%
% INPUT
% Y:  order-N tensor.
% R:  a vector [R1 R2 ... RN] or scalar if R1 = ...= RN
% opts: parameter for the decomposition
%   .tol :  tolerence (1e-6)
%   .maxiters        (50)
%   .init:   see settings in ntd_init
%   
% OUTPUT
% T : t-tensor 
%   T.core: a core tensor of size R1 x R2 x R3
%   T.U{1}, ..., T.U{N}: N factor matrices of size In x Rn
% fitarr:   array of fits and iterations.
%
% EXAMPLE
%   Y = rand(10,12,13); 
%   R = [3 4 5];
%   opts = ntd_hals3;opts.init = 'nvec';opts.maxiters = 100;
%   [T,fit] = ntd_hals3(Y,R,opts);
%   plot(fit(:,1),1-fit(:,2));xlabel('Iteration');ylabel('Relative Error')
% 
% REF: 	
% Anh Huy Phan, Andrzej Cichocki, Extended HALS algorithm for nonnegative
% Tucker decomposition and its applications for multiway analysis and
% classification, Neurocomputing, vol. 74, 11, pp. 1956-1969, 2011 

% 08/2010
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% April 2013, optimized codes for sparse tensors


if ~exist('opts','var'), opts = struct; end
opts = parseInput(opts);
if nargin == 0
    T = opts; return
end

fprintf('\n HALS NTD:\n');

% Extract number of dimensions and norm of Y.
N = ndims(Y);In = size(Y);


if numel(R) == 1
    R = R(ones(1,N));
end

normY = norm(Y);

%% Set up and error checking on initial guess for U.
[A,G] = ntd_init(Y,R,opts);
G = tensor(G);
%%
fprintf('\HALS NTD:\n');
% Compute approximate of Y
AtA = cellfun(@(x)  x'*x, A,'uni',0);fit = inf;
fitarr = [];

%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(0.001)
    fitold = fit;
    % Update Factor
    for n = 1: N
        YtA = ttm(Y,A,-n,'t');
        YtAn = tenmat(YtA,n);
        
        Gn = tenmat(G,n);
        YtAnG = YtAn * Gn';
        
        GtA = full(ttm(G,AtA,-n));
        GtAn = tenmat(GtA,n);
        B = Gn * GtAn';
        for r = 1:R(n)
            A{n}(:,r) = YtAnG(:,r) - A{n}(:,[1:r-1 r+1:end]) * B([1:r-1 r+1:end],r);
            A{n}(:,r) = max(1e-10,A{n}(:,r)/B(r,r));
        end
        ellA = sqrt(sum(A{n}.^2,1));
        G = ttm(G,diag(ellA),n);
        A{n} = bsxfun(@rdivide,A{n},ellA);
        AtA{n} = A{n}'*A{n};
    end
%     G = G.*full(ttm(Y,A,'t'))./ttm(G,AtA);   % Frobenius norm
    
    for jgind = 1:prod(R)
        jgsub = ind2sub_full(R,jgind);
        va = arrayfun(@(x) A{x}(:,jgsub(x)),1:N,'uni',0);
        Ava = arrayfun(@(x) AtA{x}(:,jgsub(x)),1:N,'uni',0);
        ava = arrayfun(@(x) AtA{x}(jgsub(x),jgsub(x)),1:N);
        
        gjnew = max(eps, ttv(Y,va) - ttv(G,Ava) + G(jgsub) * prod(ava));
        G(jgind) = gjnew;
    end
   
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
    end
end
%% Compute the final result
T = ttensor(G, A);

if nargout >=2
    output.NoIters = iter;
    output.Fit = fitarr;
end

end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ttensor')||ismember(x(1:4),{'rand' 'nvec' 'fibe' 'nmfs'})));
param.addOptional('maxiters',50);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addParamValue('lsmooth',0);
param.addParamValue('lortho',0);
param.addParamValue('lsparse',0);
param.addParamValue('alsinit',1);

param.parse(opts);
param = param.Results;
end

% function Gr = extractsubtensor(G,n,r)
% patt = 'G(';
% for k = 1:n-1
%     patt = [patt ':,'];
% end
% patt = [patt 'r,'];
% for k = n+1:ndims(G)
%     patt = [patt ':,'];
% end
% patt(end) = ')';
% patt = [patt ';'];
% Gr = eval(patt);
% 
% 
% end