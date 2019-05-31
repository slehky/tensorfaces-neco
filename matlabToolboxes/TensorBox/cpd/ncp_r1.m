function [P,out] = ncp_r1_cj(Y,opts)
% A nonlinear conjugate gradient algorithm for the Best nonnegative rank-1
% tensor approximation s
% 
% Phan Anh Huy 2017

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
N = ndims(Y);
 
%% initialization
Uinit = ncp_init(Y,1,param); u = Uinit;

%% Normalization and adjust the rank-1 to achieve the optimal norm 
gamma_ = cellfun(@(x) x'*x,u,'uni',1);

% optimal lambda 
lda = ttv(Y,u)/prod(gamma_);
if lda<0
    u{1} = -u{1};
    lda = -lda;
end
alpha = (lda*prod(gamma_))^(1/N);

% normalize u so that un^T * un = alpha
for n = 1:N
    u{n} = u{n}/sqrt(gamma_(n))*sqrt(alpha);
end


%% Output
if param.printitn ~=0
    fprintf('\Best nonnegative Rank-One tensor approximation\n');
end

N = ndims(Y);
err = 0;
 
g = cell(N,1);
normY = norm(Y);
normY2 = normY^2;
for kiter = 1:param.maxiters
    
    % balance normlization 
    gamma_ = cellfun(@(x) x'*x,u,'uni',1); % gamma = alpha
 
    gamma = prod(gamma_);
    lda = ttv(Y,u)/gamma;
    if lda<0
        u{1} = -u{1}; lda = -lda;
    end
    
    alpha = gamma^(1/N/2) * lda^(1/N);
    alpha2 = alpha^2;
    for n = 1:N
        u{n} = u{n}/sqrt(gamma_(n)) * alpha;
    end
    % gamma = lda^2*gamma;
    
    
    gamma_ns1 = alpha2^(N-1);
    
    %% find the optimal step size eta un <- un + eta * gn
    % change the parameters, u+ = u^2
    % then update up instead of u
    up = cellfun(@(x) sqrt(x),u,'uni',0);
    
    % compute the gradient of the objective function wrt u{n}
    % min f(u) = | Y - u+(1) o u+(2) o ... o u+(N)|_F^2
    
    for n = N:-1:1
        tn = double(ttv(Y,u,-n));
        g{n} = -4*up{n}.*(tn - u{n}*gamma_ns1);
    end 
    
    % Find the obtimal step size eta
    % u+ = u^2 = ([u -g] [1; eta])^2
    % 
    %  y = f(u) = |Y|_F^2 + prod_n (u+(n)^T * u+(n)) ...
    %             - 2 <Y,u+(1) o u+(2) o ... o u+(N)>
    %    = |Y|_F^2 + prod_n  ([u -g]'*[u -g]) x [1;eta]^2 - 
    %           - 2 * W x ([1;eta]^(2N)       
    % 
    
    W = ttm(Y,cellfun(@(x,y) [x.^2 -x.*y -x.*y y.^2],up,g,'uni',0),'t');
    f2 = gen_poly_kron_1x(W(:),2*N);
    
    for n = 1:N
        
        Wn = [up{n}.^2 -up{n}.*g{n} -up{n}.*g{n} g{n}.^2];
        Wn = Wn'*Wn;
        if n == 1
            f1 = gen_poly_kron_1x(Wn(:),4);
        else
            fn = gen_poly_kron_1x(Wn(:),4);
            f1 = conv(f1,fn);
        end
    end
    fcost = f1;
    fcost(2*N+1:end) = fcost(2*N+1:end) - 2*f2;
      
    % find eta which minimises the fcost 
    df = polyder(fcost);
    etas = roots(df);
    % eta should be in [0 1/alpha^(N-1)]
    etas = etas(abs(imag(etas))<1e-8);
%     eta = etas(0<=etas);
%     eta = etas((0<=etas)&(etas<1/alpha^(N-1)));
%     eta = [eta; 0 ;1/alpha^(N-1)];
    eta = [0 ; etas];
    fcost_etas = polyval(fcost,eta);
    [fcost_eta,ietas] = min(fcost_etas);
    eta = eta(ietas);
     
     
    unew = up;
    for n = 1:N
        unew{n} = up{n} - g{n}*eta;
    end
    up = unew;
    u = cellfun(@(x) x.^2,up,'uni',0); 
     
    % relative error  
    %err(kiter) = norm(Y-full(ktensor(u)));
    err(kiter) = sqrt(normY2 + fcost_eta)/normY;
     
    fprintf('%d err %d \n',kiter,err(kiter))
    if (kiter>1) && abs(err(kiter)-err(kiter-1))< param.tol
        break
    end
end

P = ktensor(u);
out.Fit = [(1:numel(err))' 1-err(:)];


end

 

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);

param.addOptional('normX',[]);


param.parse(opts);
param = param.Results;

end



%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function Uinit = ncp_init(X,R,param)
% Set up and error checking on initial guess for U.
N = ndims(X);
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init(:);
        Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
            error('Wrong Initialization');
        end
    else % small iteratons to find best initialization
        normX = norm(X);
        bestfit = 0;Pbest = [];
        cp_fun = @ncp_fLM;
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || isa(initk,'ktensor') || ... 
                    (ischar(initk)  && ismember(initk(1:4),{'fibe' 'rand' 'nvec'}))  % multi-initialization
                
                if ischar(initk)
                    cprintf('blue','Init. %d - %s',ki,initk)
                else
                    cprintf('blue','Init. %d - %s',ki,class(initk))
                end
                
                
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                P = cp_fun(X,R,initparam);
                fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
                if fitinit > bestfit
                    Pbest = P;
                    bestfit = fitinit;
                end
            end
        end
        cprintf('blue','Choose the best initial value.\n')
        Uinit = Pbest.U;
        Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
    end
    
elseif isa(param.init,'ktensor')
    Uinit = param.init.U;
    Uinit{end} = Uinit{end} * diag(param.init.lambda);
    Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
        error('Wrong Initialization');
    end    
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = rand(size(X,n),R);
    end
elseif strcmp(param.init(1:4),'nvec')
    Uinit = cell(N,1);
    for n = 1:N
        if R <= size(X,n)
            Uinit{n} = abs(real(nvecs(X,n,R)));
        else
            Uinit{n} = rand(size(X,n),R);
        end
    end
elseif strcmp(param.init(1:4),'fibe') %fiber
    Uinit = cell(N,1);
    %Xsquare = X.data.^2;
    for n = 1:N
        Xn = double(tenmat(X,n));
        %proportional to row/column length
        part1 = sum(Xn.^2,1);
        probs = part1./sum(part1);
        probs = cumsum(probs);
        % pick random numbers between 0 and 1
        rand_rows = rand(R,1);
        ind = [];
        for i=1:R,
            msk = probs > rand_rows(i);
            msk(ind) = false;
            ind(i) = find(msk,1);
        end
        Uinit{n} = Xn(:,ind);
        Uinit{n} = bsxfun(@rdivide,Uinit{n},sqrt(sum(Uinit{n}.^2)));
    end
else
    error('Invalid initialization');
end
end