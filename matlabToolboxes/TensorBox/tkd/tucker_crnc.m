function [T,Uinit,out] = tucker_crnc(X,R,opts)
% Crank-Nicholson algorithm for orthogonal Tucker decomposition
%
%   T = tucker_crnc(X,R) computes the best rank(R1,R2,..,Rn)
%   approximation of tensor X, according to the specified dimensions
%   in vector R.  The input X can be a tensor, sptensor, ktensor, or
%   ttensor.  The result returned in T is a ttensor.
%
%   T = tucker_crnc(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init' - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations {1}
%
%   [T,U0] = tucker_crnc(...) also returns the initial guess.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   T = tucker_crnc(X,2);        %<-- best rank(2,2,2) approximation
%   T = tucker_crnc(X,[2 2 1]);  %<-- best rank(2,2,1) approximation
%   T = tucker_crnc(X,2,'dimorder',[3 2 1]);
%   T = tucker_crnc(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),[]}; %<-- Initial guess for factors of T
%   T = tucker_crnc(X,2,'dimorder',[3 2 1],'init',U0);
%
% The function uses the Matlab Tensor toolbox.
%   See also: tucker_als
%
% The zero row and columns needs to be removed from the data.
%
% Ref:
%  Anh-Huy Phan, Andrzej Cichocky and Petr Tichavsky, On fast algorithm for
%  orthogonal Tucker decomposition, ICASSP 2014, pp. 6766 - 6770.
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors.
%
% This algorithm is a part of the TENSORBOX, 2013.


% Fill in optional variable
if ~exist('opts','var'), opts = struct; end

% Extract number of dimensions and norm of X.
try
    N = ndims(X);
    normX = norm(X);
catch
    N = [];
end

params = parseInput(opts,N);
if nargin == 0
    T = params; return
end

% Crank-Nicholson parameters
crnc_param = params.crnc_param;

%% Copy from params object
fitchangetol = params.tol;
maxiters = params.maxiters;
dimorder = params.dimorder;
if isempty(dimorder)
    dimorder = 1:N;
end
init = params.init;
printitn = params.printitn;

if numel(R) == 1
    R = R(ones(1,N));
end

%% Error checking
% Error checking on maxiters
if maxiters < 0
    error('OPTS.maxiters must be positive');
end

% Error checking on dimorder
if ~isequal(1:N,sort(dimorder))
    error('OPTS.dimorder must include all elements from 1 to ndims(X)');
end

%% Set up for iterations - initializing U and the fit.
Uinit = tucker_init(X,R,params);
U = Uinit;
fit = 0;

if printitn > 0
    fprintf('\nCrank-Nicholson algorithm for Tucker decomposition:\n');
end
normX2 = normX^2;
errorarr = [];

if strcmp(crnc_param.stepsize,'combine')
    crnc_param2 = crnc_param;
    crnc_param2.maxiter = 3;
end
hooicnt = [];
%% Main Loop: Iterate until convergence
iter = 1;
while iter <=maxiters
    if (params.crnc_param.refine_stepsize~=0) && mod(iter,params.crnc_param.refine_stepsize) ==0
        crnc_param.refine_stepsize = 1;
    else
        crnc_param.refine_stepsize= 0;
    end
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    for n2 = dimorder(1:end)
        if n2 == N
            nn = [1 n2];
        else
            nn = [n2+1 n2];
        end
        
        narr = setdiff(1:N,nn);
        K = ttm(X,U,narr,'t');
        
        for n = nn
            narr2 = setdiff(nn,n);
            Zn = ttm(K,U,narr2,'t');
            
            Cn = tenmat(Zn,n);Cn = double(Cn*Cn');
            
            switch crnc_param.stepsize
                case {'bb'  'barzilai-borwein' 'Barzilai-Borwein'}
                    [U{n},out] = crank_nicholson_bb(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    
                case 'poly8'
                    [U{n},out] = crank_nicholson_optstepsize(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    %U{n} = OptStiefelGBB_optstepsize_fast(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    if strcmp(out.msg,'empty')
                        [U{n},out] = crank_nicholson_bb(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    end
                    
                case 'taylor-3'
                    [U{n},out] = crank_nicholson_optstepsize2(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    %U{n} = OptStiefelGBB_optstepsize_fast(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    if strcmp(out.msg,'empty')
                        [U{n},out] = crank_nicholson_bb(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    end
                    
                case 'combine'
                    [U{n},out] = crank_nicholson_optstepsize(U{n}, @fastfevalTucker, crnc_param2,Cn,normX2);
                    [U{n},out] = crank_nicholson_bb(U{n}, @fastfevalTucker, crnc_param,Cn,normX2);
                    
            end
        end
    end
    
    % Compute fit
    normresidual = real(sqrt(2*out.F));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);
    
    if mod(iter,printitn)==0
        fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', iter, fit, fitchange);
    end
    errorarr = [errorarr 1-fit];
    
    
    % Check for convergence
    if (iter > 1) && ((fitchange < fitchangetol) || (1-fit < params.min_relerror))
        if params.hooi_boostup
            % Call HOOI if there is no any improvement on CrNc
            iter = iter+1;
            hooicnt = [hooicnt iter];
            for n = dimorder(1:end)
                Zn = ttm(X,U,[1:n-1 n+1:N],'t');
                U{n} = nvecs(Zn,n,R(n));
            end
            Zn = ttm(Zn,U,N,'t');
            normresidual = real(sqrt(normX2 - norm(Zn)^2));
            fit = 1-normresidual/normX;
            errorarr = [errorarr normresidual/normX];
            fitchange = abs(errorarr(end)-errorarr(end-2));
        end
        if (fitchange < fitchangetol)
            break;
        end
    end
    iter = iter+1;
end
if (size(Zn,N) ~= R(N))
    Zn = ttm(Zn, U, N, 't');
end
% Zn = ttm(X, U, 't');
T = ttensor(Zn, U);
out  = struct('Error',errorarr,'hooicnt', hooicnt);
end



%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function params = parseInput(opts,N)
if (nargin < 2)
    N = [];
end
%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.KeepUnmatched = true;
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParamValue('dimorder',1:N);
params.addParamValue('init', 'nvec', @(x) (iscell(x) || ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth'})));
params.addParamValue('printitn',1,@isscalar);

params.addParamValue('normX',[]);
params.addParamValue('min_relerror',0);
params.addParamValue('hooi_boostup',0);


% Crank-Nicholson parameters
crnc_param = inputParser;
crnc_param.KeepUnmatched = true;
crnc_param.addParamValue('maxiters',10);
crnc_param.addParamValue('gtol',1e-5);
crnc_param.addParamValue('xtol',1e-5);
crnc_param.addParamValue('ftol',1e-5);
crnc_param.addParamValue('tau',1e-3);
crnc_param.addParamValue('rho',1e-4); % parameters for control the linear approximation in line search
crnc_param.addParamValue('eta',.2); % factor for decreasing the step size in the backtracking line search
crnc_param.addParamValue('gamma',.85); % updating C
crnc_param.addParamValue('nt',5);
crnc_param.addParamValue('refine_stepsize',false);
% crnc_param.addParamValue('iscomplex',0);
crnc_param.addParamValue('stepsize', 'barzilai-borwein', @(x) (ismember(x,{'bb' 'barzilai-borwein' 'poly8' 'combine' 'taylor-3'})));
% optimal step size or using the Barzilai-Borwein method

crnc_opts = struct;
if isfield(opts,'crnc_param')
    crnc_opts = opts.crnc_param;
end
crnc_param.parse(crnc_opts);
crnc_param = crnc_param.Results;
params.addParamValue('crnc_param',crnc_param);

params.parse(opts);
params = params.Results;
end

function [F,CU,UCU,G,GG] = fastfevalTucker(Un,Cn,normX2,GGp,CUp,CGp,UCUp,GCGp,Omega,tau)
% Un: n-th factor
% Zn = mode-n unfolding of X x_{-n} {U}
% Cn = Zn*Zn';
% F cost value
% G: gradient of cost with respect to Un
if nargin<3
    normX2 = 0;
end
if nargin<4
    CU = Cn*Un;
    UCU = Un'*CU;
else
    CU = - CUp + (2*CUp - tau * CGp)*Omega;
    W = 2*Omega -eye(size(Omega));
    UCU = W*UCUp*W + 2*tau*GGp*Omega*W + tau^2 * Omega * GCGp * Omega;
end

F = (normX2-trace(UCU))/2;%-sum(sum((Yn - Vn*Zn).^2))/2;

if nargout >=4
    G = - CU + Un * UCU;
end
% Precomputing G'*G instead of direct G'*G since G'*U may not be
% exactly zero.
if nargout >=5
    GG = CU'*CU - UCU*UCU;
end
end

%%
function [X, out]= crank_nicholson_bb(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% Update factor matrix U == X in the Tucker decomposition using the
% Barzilai-Borwein method to choose step size
%
%   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
% See also OptStiefelGBB
% Reference:
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
%-------------------------------------------------------------------------
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('xtol',1e-6,@isnumeric);
param.addOptional('gtol',1e-6,@isnumeric);
param.addOptional('ftol',1e-12,@isnumeric);
% parameters for control the linear approximation in line search
param.addOptional('rho',1e-4,@isnumeric);
param.addOptional('eta',.2,@isnumeric);
param.addOptional('gamma',.85,@isnumeric);
param.addOptional('tau',1e-3,@isnumeric);
param.addOptional('nt',5,@isnumeric);
param.addOptional('maxiters',100,@isnumeric);
param.addOptional('refine_stepsize',false);

param.parse(opts);
opts = param.Results;

%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
rho  = opts.rho;
eta   = opts.eta;
gamma = opts.gamma;
nt = opts.nt;   crit = ones(nt, 3);

[n, k] = size(X);

%% Initial function value and gradient
% prepare for iterations
[F,CU,UCU,G,GG] = feval(fun, X , varargin{:});  out.nfe = 1;

nrmG  = sqrt(trace(GG));%nrmG = norm(G,'fro');
Q = 1; Cval = F;  tau = opts.tau;

X0 = X; F0 = F;
C = varargin{1};normX2 = varargin{2};
%% main iteration
for itr = 1 : opts.maxiters
    XP = X;     FP = F;   GP = G;  GGP = GG; CUP = CU;
    UCUP = UCU;
    %     CGP = C*G;
    %UCUP = X'*CU;
    UCGP = CU'*G;
    
    XX = X'*X;
    X1 = XP * (eye(k)-2*XX);
    GX = G'*X;
    
    deriv = rho*nrmG^2; %deriv
    nls = 1;
    while 1
        Omega = inv(eye(k)+tau^2/4*GGP);
        X = X1 + (2*XP - tau * GP)*Omega*(XX + tau/2*GX);
        
        %         X = -XP + (2*XP - tau * GP)*Omega;
        %         F = feval(fun, X, varargin{:});
        UCU = - UCUP + (2*UCUP - tau * UCGP)*Omega;
        % %         CU = - CUP + (2*CUP - tau * CGP)*Omega;
        %         UCU = X'*CU;
        F = (normX2-trace(UCU))/2; % Total cost 1/2\|Y - Yhat \|_F^2
        
        out.nfe = out.nfe + 1;
        if F< FP || nls >= 5 %if F <= Cval - tau*deriv || nls >= 5
            break;
        end
        tau = eta*tau; nls = nls+1;
    end
    % Update gradient
    CU = C*X;UCU = X'*CU;
    %     CU = - CUP + (2*CUP - tau * CGP)*Omega;
    G = - CU + X * UCU;
    GG = G'*G;
    %GG = CU'*CU - UCU*UCU;
    %     [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
    
    %% Refine step size
    if opts.refine_stepsize~=0
        F1 = F;
        GCG = G'*C*G;
        
        [uo,so] = eig(max(GG,GG'));so = real(diag(so));
        d1 = diag(uo'*UCU*uo);
        d2 = diag(uo'*GCG*uo);
        d = d2-so.*d1;
        
        % Total cost D = 1/2*\|Y - Yhat\|_F^2 = 1/2(\|Y\|_F^2 - 1^T d1 - f_tau(tau))
        f_tau = @(x) -x*sum((-x.^2*so.^2+2*x*d+4*so)./(4+x.^2*so).^2);
        
        taus = [];
        % Approximate denumerator of gx by a Taylor order -5 and find tau
        %taus = fastsolvestepsize(so,d);
        
        sd = sum(d);sso2 = sum(so.^2); sso = sum(so);
        % Taylor-3
        tau2 = real(2/9*(sd + sqrt(sd^2 + 9 *sso*sso2))/sso2);
        if tau2>1
            tau2 = [];
        end
        taus = real([taus; tau2;tau]);
        
        % Choose the root which yields the lowest cost function
        Fkt = zeros(numel(taus),1);
        for kt = 1:numel(taus)-1
            Fkt(kt) = f_tau(taus(kt));
        end
        Fkt = (normX2 -sum(d1) + 8*Fkt)/2;  % Cost function 1/2*\|Y-Yhat\|_F^2
        Fkt(end) = F1;
        [F2,tid] = min(Fkt);
        tau = taus(tid(1));
        
        %     gx = @(x) (x^4*so.^3 - 24*x^2*(so.^2) + 16*so -4*x*d.*(x^2*so - 4))'*(1./(4+x^2*so).^3);
        %     tau = fminsearch(f_tau, tau);
        %     tau = fminbnd(f_tau,0, 5*tau);
        %     tau = fzero(gx,tau);
        tau = fminbnd(f_tau,0,5*tau);
        F2 = (normX2 -sum(d1) + 8*f_tau(tau))/2;
        if F2<F1
            Omega = inv(eye(k)+tau.^2/4*GG);
            XX = X'*X;
            X1 = X * (eye(k)-2*XX);
            GX = G'*X;
            X = X1 + (2*X - tau * G)*Omega*(XX + tau/2*GX);
            
            %X = -X + (2*X - tau * G)*Omega;
            [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
        end
    end
    
    %%
    XG = X'*G;
    flag = norm(XG,'fro')/k > 1e-3;
    %XX = X'*X;
    %flag = norm(XX-eye(k),'fro')/k > 1e-5;
    if flag % GramSchmidt
        [u,s,v] = svd(X,0);X = u*v';
        %[v,s] = eig(XX); X = X*v*diag(1./sqrt(diag(s)))*v';
        [F,CU,foe,G,GG] = feval(fun, X, varargin{:});
    end
    
    nrmG  = sqrt(trace(GG)); %nrmG  = norm(G, 'fro');
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    FDiff = abs(FP-F)/(abs(FP)+1);
    
    Y = G - GP;     SY = abs(sum(sum(S.*Y)));
    if mod(itr,2)==0; tau = (sum(sum(S.*S))+eps)/(SY+eps);
    else tau  = (SY+eps)/(sum(sum(Y.*Y))+eps); end
    tau = max(min(tau, 1e20), 1e-20);
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    
    
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
end

XX = X'*X;
flag = norm(XX-eye(k),'fro')/k > 1e-7;
if flag % GramSchmidt
    [u,s,v] = svd(X,0);X = u*v';
    %[v,s] = eig(XX); X = X*v*diag(1./sqrt(diag(s)))*v';
    F = feval(fun, X, varargin{:});
end
out.F = F;
if F > F0
    X = X0;out.F = F0;
end
end

%%
function [X, out]= crank_nicholson_optstepsize(X, fun, opts, varargin)
% Update factor matrix of the TUcker decomposition where step size is
% chosen as positive root of a polynomial of degree-8.
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('xtol',1e-6,@isnumeric);
param.addOptional('gtol',1e-6,@isnumeric);
param.addOptional('ftol',1e-12,@isnumeric);

param.addOptional('tau',1e-3,@isnumeric);
param.addOptional('nt',5,@isnumeric);
param.addOptional('maxiters',1000,@isnumeric);

param.parse(opts);
opts = param.Results;

%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
nt = opts.nt;   crit = ones(nt, 3);

[n, k] = size(X);
C = varargin{1};
normX2 = varargin{2};

%% Initial function value and gradient
% prepare for iterations
[F,CU,UCU,G,GG] = feval(fun, X , varargin{:});  out.nfe = 1;
F0 = F;X0 = X;
out.msg = '';tau = opts.tau;
%% main iteration
for itr = 1 : opts.maxiters
    XP = X;     FP = F;   GP = G;  GGP = GG;CUP = CU; UCUP = UCU;%dtXP = dtX;
    
    CG = C*G; GCGP = G'*CG;
    
    
    % FInd step size as positive roots of a degree-8 polynomial
    [uo,so] = eig((GGP+GGP')/2);so = real(diag(so));
    d1 = diag(uo'*UCUP*uo);
    d2 = diag(uo'*GCGP*uo);
    d = d2-so.*d1;
    
    % Cost to find eta
    %     f_tau = @(x) -(((4-x.^2*so)./(4+x.^2*so)).^2).'*d1  ...
    %     - 2*x*(((4*(4-x.^2*so)./(4+x.^2*so).^2)).'*so)  ...
    %     - x.^2*((16./(4+x.^2*so).^2).'*d2);
    
    % Total cost D = 1/2*\|Y - Yhat\|_F^2 = 1/2(\|Y\|_F^2 - 1^T d1 - f_tau(tau))
    f_tau = @(x) -x*sum((-x.^2*so.^2+2*x*d+4*so)./(4+x.^2*so).^2);
    
    
    % T = GCGP - GGP*UCUP;
    % gx2 = @(x) trace(inv(eye(k)+x^2/4*GGP)^3*(x^4*GGP^3 -24*x^2*GGP^2 - 4*x^3*GGP*T + 16*x*T+16*GGP))/64;
    
    % Approximate denumerator of gx by a Taylor order -5 and find tau
    taus = fastsolvestepsize(so,d);
    
    sd = sum(d);sso2 = sum(so.^2); sso = sum(so);
    tau2 = real(2/9*(sd + sqrt(sd^2 + 9 *sso*sso2))/sso2);
    if tau2>1
        tau2 = [];
    end
    
    % % Taylor order 2
    %     p2 = [sum(so.^3)/16 -1/4*(so'*d) -3/2*sum(so.^2) sum(d)  sum(so)];
    %     taus2 =roots(p2);
    taus = real([taus; tau2;tau]);
    
    % %Taylor order 3
    % p2 = [-3/64*sum(so.^4) 3/16*so.^2'*d  19/16*sum(so.^3)  -so'*d -9/4*sum(so.^2) sum(d)  sum(so)];
    % taus =roots(p2);
    
    %     if isempty(taus)
    %         taus = tau;
    % %         out.msg = 'empty'; % cannot find optimal step size as root of poly-8, switch to using the BB method
    % %         break
    % %     else
    %
    %     end
    
    % Choose the root which yields the lowest cost function
    Fkt = zeros(numel(taus),1);
    for kt = 1:numel(taus)
        Fkt(kt) = f_tau(taus(kt));
    end
    Fkt = (normX2 -sum(d1) + 8*Fkt)/2;  % Cost function 1/2*\|Y-Yhat\|_F^2
    %     Fkt = Fkt(Fkt<=F);
    
    [F,tid] = min(Fkt);
    tau = taus(tid(1));
    
    
    tau = fminbnd(f_tau,0,10*tau);
    Omega = inv(eye(k)+tau.^2/4*GGP);
    
    
    % Refine step size tau
    %Gradient of f_tau w.r.t tau used to refine the step size
    %gx = @(x) (x^4*so.^3 - 24*x^2*(so.^2) + 16*so -4*x*d.*(x^2*so - 4))'*(1./(4+x^2*so).^3);
    %     tau = fminsearch(f_tau, tau);
    %tau = fminbnd(f_tau,0, 2*tau);
    %tau = fzero(gx,tau);
    
    % Update X
    
    X = -XP + (2*XP - tau * GP)*Omega;
    [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
    %     % Update Gradient
    %     CU = - CUP + (2*CUP - tau * CG)*Omega;
    %     W = 2*Omega -eye(size(Omega));
    %     UCU = W*UCUP*W + 2*tau*W*GGP*Omega + tau^2 * Omega * GCGP * Omega;
    %     G = - CU + X * UCU;
    %     % Precomputing G'*G instead of direct G'*G since G'*U may not be
    %     % exactly zero.
    %     GG = CU'*CU - UCU*UCU;
    
    XX = X'*X;
    flag = norm(XX-eye(k),'fro')/k > 1e-5;
    if flag
        %X = MGramSchmidt(X);
        [v,s] = eig(XX); X = X*v*diag(1./sqrt(diag(s)))*v';
        [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
    end
    
    nrmG  = sqrt(trace(GG)); %nrmG  = norm(G, 'fro');
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    FDiff = abs(FP-F)/(abs(FP)+1);
    
    Y = G - GP;     SY = abs(sum(sum(S.*Y)));
    if mod(itr,2)==0; tau = (sum(sum(S.*S))+eps)/(SY+eps);
    else tau  = (SY+eps)/(sum(sum(Y.*Y))+eps); end
    tau = max(min(tau, 1e20), 1e-20);
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    
    
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
        if itr <= 2
            ftol = 0.1*ftol;xtol = 0.1*xtol;gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
end
out.F = F;

if F > F0
    X = X0;out.F = F0;
end

end

%% ORDER-3 TAYLOR APPROXIMATION OF F(TAU
%%
function [X, out]= crank_nicholson_optstepsize2(X, fun, opts, varargin)
% Update factor matrix of the TUcker decomposition where step size is
% chosen as positive root of a polynomial of degree-8.
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('xtol',1e-6,@isnumeric);
param.addOptional('gtol',1e-6,@isnumeric);
param.addOptional('ftol',1e-12,@isnumeric);

param.addOptional('tau',1e-3,@isnumeric);
param.addOptional('nt',5,@isnumeric);
param.addOptional('maxiters',1000,@isnumeric);

param.parse(opts);
opts = param.Results;

%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
nt = opts.nt;   crit = ones(nt, 3);

[n, k] = size(X);
C = varargin{1};
normX2 = varargin{2};

%% Initial function value and gradient
% prepare for iterations
[F,CU,UCU,G,GG] = feval(fun, X , varargin{:});  out.nfe = 1;
F0 = F;X0 = X;
out.msg = '';
%% main iteration
for itr = 1 : opts.maxiters
    XP = X;     FP = F;   GP = G;  GGP = GG;CUP = CU; UCUP = UCU;%dtXP = dtX;
    
    CG = C*G; GCGP = G'*CG;
    
    
    % Find step size
    [uo,so] = eig(GGP);so = diag(so);
    d1 = diag(uo'*UCUP*uo);
    d2 = diag(uo'*GCGP*uo);
    d = d2-so.*d1;
    
    sd = sum(d);sso2 = sum(so.^2); sso = sum(so);
    
    tau = 2/9*(sd + sqrt(sd^2 + 9 *sso*sso2))/sso2;
    % Refine step size tau
    %Gradient of f_tau w.r.t tau used to refine the step size
    gx = @(x) (x^4*so.^3 - 24*x^2*(so.^2) + 16*so -4*x*d.*(x^2*so - 4))'*(1./(4+x^2*so).^3);
    %tau = fminsearch(f_tau, tau);
    %tau = fminbnd(f_tau,0, 2*tau);
    tau = fzero(gx,tau);
    
    % Update X
    Omega = inv(eye(k)+tau.^2/4*GGP);
    X = -XP + (2*XP - tau * GP)*Omega;
    %     [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
    %     % Update Gradient
    CU = - CUP + (2*CUP - tau * CG)*Omega;
    W = 2*Omega -eye(size(Omega));
    UCU = W*UCUP*W + tau*W*GGP*Omega + tau^2 * Omega * GCGP * Omega;
    G = - CU + X * UCU;
    %     % Precomputing G'*G instead of direct G'*G since G'*U may not be
    %     % exactly zero.
    GG = CU'*CU - UCU*UCU;
    
    XX = X'*X;
    flag = norm(XX-eye(k),'fro')/k > 1e-5;
    if flag
        %X = MGramSchmidt(X);
        [v,s] = eig(XX); X = X*v*diag(1./sqrt(diag(s)))*v';
        [F,CU,UCU,G,GG] = feval(fun, X, varargin{:});
    end
    
    nrmG  = sqrt(trace(GG)); %nrmG  = norm(G, 'fro');
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    FDiff = abs(FP-F)/(abs(FP)+1);
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    
    out.F = F;
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
        if itr <= 2
            ftol = 0.1*ftol;xtol = 0.1*xtol;gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
end

XX = X'*X;
flag = norm(XX-eye(k),'fro')/k > 1e-5;
if flag
    %X = MGramSchmidt(X);
    [v,s] = eig(XX); X = X*v*diag(1./sqrt(diag(s)))*v';
    F = feval(fun, X, varargin{:}); out.F = F;
end


end


%%
function xs = fastsolvestepsize(so, d)
% find step size eta as root of a degree 8 poly

% [uo,so] = eig(GG);so = diag(so);
% d = diag(uo'*GCG*uo)-so.*diag((uo'*XCX*uo));
% Fast construction of polynomial p


% Fast construction of polynomial p
px(9) = -sum(so);
px(8) = -sum(d);
sop = so.^2;
px(7) = 9/4*sum(sop);
px(6) = so'*d;
px(4) = -9/16*sop'*d;
sop = sop.*so;
px(5) = -25/16*sum(sop);
px(2) = 3/32* so.^3'*d;
sop = sop.*so;
px(3) = 39/64*sum(sop);
sop = sop.*so;
px(1) = -3/128*sum(sop);

xs = roots(px);
xs = xs(imag(xs)==0);
xs = sort(xs);
xs = xs(xs>=0);
end