function [T,fitarr] = ntd_QALS(Y,R,opts)
% ALS algorithms for NTF based on Nonnegative Quadratic Programming
% Copyright by Phan Anh Huy, 08/2010
% Ref: Novel Alternating Least Squares Algorithm for
%      Nonnegative Matrix and Tensor Factorizations
%      Anh Huy Phan, Andrzej Cichocki, Rafal Zdunek, and Thanh Vu Dinh
%      ICONIP 2010

defoptions = struct('tol',1e-6,'maxiters',50,'init','random',...
    'ellnorm',2,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,...
    'fixsign',0,'rowopt',1);
if ~exist('opts','var')
    opts = struct;
end
opts = scanparam(defoptions,opts);

% Extract number of dimensions and norm of Y.
N = ndims(Y);
normY = norm(Y);

if numel(R) == 1
    R = R(ones(1,N));
end
if numel(opts.lda_ortho) == 1
    opts.lda_ortho = opts.lda_ortho(ones(1,N));
end
if numel(opts.lda_smooth) == 1
    opts.lda_smooth = opts.lda_smooth(ones(1,N));
end
In = size(Y);
%% Set up and error checking on initial guess for U.
[A,G] = ntd_initialize(Y,opts.init,opts.orthoforce,R);
G = tensor(G);
%%
fprintf('\nLocal NTD:\n');
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
end


function Gr = extractsubtensor(G,n,r)
patt = 'G(';
for k = 1:n-1
    patt = [patt ':,'];
end
patt = [patt 'r,'];
for k = n+1:ndims(G)
    patt = [patt ':,'];
end
patt(end) = ')';
patt = [patt ';'];
Gr = eval(patt);


end