function [Uinit,Ginit] = ntd_init(X,R,opts)
% Initialization for Tucker Decomposition
% TENSORBOX implements several simple and efficient methods to initialize
% factor matrices.
%   - (random) Random initialization which is always the simplest and fastest, but
%    not the most efficient method.
%
%   - (nvec) SVD-based initialization which initializes factor matrices by
%    leading singular vectors of mode-n unfoldings of the tensor. The
%    method is similar to higher order SVD or multilinear SVD.
%    This method is often better than using random values in term of
%    convergence speed.
%    However, this method is less efficient when factor matrices comprise
%    highly collinear components.
%
%   - (fiber) Using fibers selected from the data for initialization is also suggested.
%
%   - Multi-initialization with some small number of iterations are often
%   performed. The component matrices with the lowest approximation
%   error are selected.
%
%   - (orth) Orthogonal factor matrices
%
% Factor matrices can be initialized outside algorithms using the routine
% "tucker_init"
%
%   opts = tucker_init;
%   opts.init = 'nvec';
%   [Uinit,Ginit] = tucker_init(X,R,opts);
%
% then passed into the algorithm through the optional parameter "init"
%   opts = tucker_als;
%   opts.init = {Uinit}; % or can directly set  opts.init = 'nvec';
%   P = tucker_als(X,R,opts);
%
% For the fLM algorithm, ALS with small runs can be employed before the main
% algorithm by setting
%   opts.alsinit = 1;
%
% See also:  tucker_als, tucker_crnc, nvecs, Nway and PLS toolbox
%
% TENSOR BOX, v1. 2012
% Copyright 2008, 2011, 2012, 2013 Phan Anh Huy.


%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ttensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth'})));
param.addOptional('alsinit',0);
param.addOptional('tucker_func',@tucker_als,@(x) isa(x,'function_handle'));

if ~exist('opts','var'), opts = struct; end
param.parse(opts);param = param.Results;
if nargin == 0
    Uinit = param; return
end

% Set up and error checking on initial guess for U.
N = ndims(X); In = size(X);
if numel(R) == 1
    R = R(ones(1,N));
end
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init;
        Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),In')) || (~all(Sz(:,2)==R(:)))
            error('Wrong Initialization');
        end
    elseif (numel(param.init) == (N+1)) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init(1:end-1);
        Ginit = param.init{end};
        Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),In')) || (~all(Sz(:,2)==R(:))) || ~isequal(size(Ginit),R)
            error('Wrong Initialization');
        end
    else % small iteratons to find the best initialization
        
        normX = norm(X);
        bestfit = 0;Pbest = [];
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || isa(initk,'tensor') || ...
                    (ischar(initk)  && ismember(initk(1:4), ...
                    {'rand' 'nvec' 'fibe' 'orth' 'nmfs'}))  % multi-initialization
                if ischar(initk)
                    cprintf('blue','Init. %d - %s\n',ki,initk)
                else
                    cprintf('blue','Init. %d - %s\n',ki,class(initk))
                end
                
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                P = param.ntd_func(X,R,initparam);
                fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
                if real(fitinit) > bestfit
                    Pbest = P;
                    bestfit = fitinit;kibest = ki;
                end
            end
        end
        cprintf('blue','Choose the best initial value: %d.\n',kibest);
        Uinit = Pbest.U;
        Ginit = Pbest.core;
    end
    
elseif isa(param.init,'ttensor')
    Uinit = param.init.U;Ginit = param.init.core;
    Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),In')) || (~all(Sz(:,2)==R(:)))
        error('Wrong Initialization');
    end
elseif ischar(param.init)
    switch param.init(1:4)
        case 'rand'
            Uinit = arrayfun(@rand,In,R,'uni',0);
            if ~param.alsinit
                Ginit = tensor(rand(R));
            end
        case {'nvec' 'eigs'}
            Uinit = cell(N,1);Ym = reshape(double(X),In(1),[]);
            
            for n = 1:N
                [u,s,v] = svds(Ym, R(n), 'L');
                
                Ym = bsxfun(@times, v,diag(s)');
                if n <N
                    Ym = reshape(Ym,In(n+1),[]);
                end
                
                Uinit{n} = u;
            end
            if ~param.alsinit
                Ginit = tensor(abs(Ym),R);
            end
            
        case 'fibe'
            Uinit = cell(N,1);
            %Xsquare = X.data.^2;
            for n = 1:N
                if isa(X,'tensor')
                    Yn = tenmat(X,n);Yn = Yn.data;
                else
                    Yn = double(sptenmat(X,n));
                end
                
                %proportional to row/column length
                part1 = sum(Yn.^2,1);
                probs = part1./sum(part1);
                probs = cumsum(probs);
                % pick random numbers between 0 and
                rand_rows = rand(R(n),1);
                ind = [];
                for i=1:R(n),
                    msk = probs > rand_rows(i);
                    msk(ind) = false;
                    ind(i) = find(msk,1);
                end
                Uinit{n} = full(Yn(:,ind));
                Uinit{n} = bsxfun(@rdivide,Uinit{n},sqrt(sum(Uinit{n}.^2)));
            end
            
            if ~param.alsinit
                Ginit = ttm(X, Uinit, 't');
            end
            
            
            
        case 'orth'
            Uinit = cell(N,1);
            for n = 1:N
                Uinit{n} = orth(randn(In(n),R(n)));
            end
            if ~param.alsinit
                Ginit = ttm(X, Uinit, 't');
            end
            
        otherwise
            error('Undefined initialization type');
    end
end

%% Powerful initialization - ALS
if param.alsinit
    for n = 1:N
        Atilde = ttm(X, Uinit, -n, 't');
        Uinit{n} = max(eps,nvecs(Atilde,n,R(n)));
    end
    Uinit = cellfun(@(x) bsxfun(@rdivide,x,sum(x)),Uinit,'uni',0);
    UtU = cellfun(@(x) inv(x'*x),Uinit,'uni',0);
    Ginit = ttm(ttm(X,Uinit,'t'),UtU);
    Ginit = max(Ginit.data,eps);
end
end