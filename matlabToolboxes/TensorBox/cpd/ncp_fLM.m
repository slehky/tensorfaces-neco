function [P,output] = ncp_lm(X,R,opts)
% Fast Damped Gauss-Newton (Levenberg-Marquard) algorithm factorizes
%       an N-way nonnegative tensor X into nonnegative factor matrices of R
%       components. 
%       The code inverts the approximate Hessian with a cost of O(NR^6) or
%       O(N^3R^6) compared to O(R^3(I1+...+IN)) in other dGN/LM algorithms.
%
% INPUT:
%   X:  N-way nonnegative data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'| ktensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'random' 'nvec'};
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%     .TraceFit: check fit values as stoping condition.
%
% OUTPUT: 
%  P:  ktensor of estimated factors
%  output:  
%      .Fit
%      .NoIters 
%
% EXAMPLE
%   X = tensor(rand([10 20 30]));  
%   opts = ncp_lm;
%   opts.init = 'nvec';
%   [P,output] = ncp_lm(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
%
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "Low Complexity Damped
% Gauss-Newton Algorithms for CANDECOMP/PARAFAC", SIAM, SIAM, Jour-
% nal on Matrix Analysis and Applications, vol. 34, pp. 126?147, 2013.
%
% [2] A. H. Phan, P. Tichavsk?y, and A. Cichocki, Fast damped Gauss-Newton
% algorithm for sparse and nonnegative tensor factorization, in ICASSP,
% 2011, pp. 1988-1991.
% 
% [3] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition, IEEE TSP, in print,
% 2013, available online at http://arxiv.org/abs/1209.3215.
%
% [4] A.-H. Phan, P. Tichavsky, A. Cichocki, "Fast Alternating LS
% Algorithms for High Order CANDECOMP/PARAFAC Tensor Factorization",
% http://arxiv.org/abs/1204.1586. 
%
% [5] P. Tichavsky and Z. Koldovsky, Simultaneous search for all modes in
% multilinear models, ICASSP, 2010, pp. 4114 ? 4117.
% 
% The function uses the Matlab Tensor toolbox.
% See also: ncp_fLMa, cp_init, cp_fastals. 
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors.
%
% This algorithm is a part of the TENSORBOX, 2012.

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

N = ndims(X); normX = norm(X);normX2 = normX^2;In = size(X);
param.normX = normX; param.normX2 = normX2;
lspar = param.sparse;
lspar(end+1:N) = lspar(1);
lspar = reshape(lspar,1,N);


%% Initialize factors U
param.cp_func = str2func(mfilename);param.nonnegative = true;
% Uinit = cp_init(X,R,param); U = Uinit;

Uinit = ncp_init(X,R,param); U = Uinit;

U = cellfun(@abs,U,'uni',0);

if param.printitn>0
    fprintf('\nLM for low rank nonnegative tensor approximation:\n');
end

% Reorder dimensions for fast computation of CP gradients
p_perm = [];
if ~issorted(In)
    [In,p_perm] = sort(In);
    X = permute(X,p_perm);
    U = U(p_perm);
end

%%
nu=2; tau=1;
ell2 = zeros(N,R);
for n = 1:N
    ell2(n,:) = sum(abs(U{n}).^2);
end
mm = zeros(N,R);
for n = 1:N
    mm(n,:) = prod(ell2([1:n-1 n+1:N],:),1);
end
mu=tau*max(mm(:));

warning off;
alpha=10*max(prod(ell2).^(1/N));   %%% initial multiplicative constant for the barrier function
dalpha=exp(100/param.maxiters*log(0.5));
% dalpha=exp(100/max(30,min(maxiters,maxiters-50))*log(0.5));
P = ktensor(U);
errf = normX2 + norm(P)^2 - 2 * innerprod(P,X) ;
err= 1/2 *errf-alpha*sum(cellfun(@(x) sum(log(x(:))),U)) ...
    + lspar * cellfun(@(x) sum(abs(x(:))) , U);
fit = 1-sqrt(errf)/normX; fitold = fit; fitarr = [1 fit];


% beta =[];
% for n = 1:N
%     beta = [beta ; lspar(n) * ones(In(n),1)];
% end


% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
ns = find(Jn<=Kn,1,'last');

%% Correlation matrices Cn
C = zeros(R,R,N);Gamman = zeros(R,R,N);
for n = 1:N
    C(:,:,n) = U{n}'*(U{n});
end

% CP-gradient X(n) * Khatri-Rao(U) except Un
Gxn = cell(N,1); 
Pmat = [];
for n = [ns:-1:1 ns+1:N]
    % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
    if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
        Gxn{n} = mttkrp(X,U,n);
    elseif isa(X,'tensor') || isa(X,'sptensor')
        if (n == ns) || (n == ns+1)
            [Gxn{n},Pmat] = cp_gradient(U,n,X);
        else
            [Gxn{n},Pmat] = cp_gradient(U,n,Pmat);
        end
    end
end

%% Main Loop: Iterate until convergence
iter = 1;iter2 = 1;
lspar0 = lspar;
while (iter <=param.maxiters) && (iter2 <= param.maxiters*10)
    iter2 = iter2+1;
    pause(.0001)
    
    U1 = U;
    
    [d Rr lspar alpha] = LMp_stepsize(X,U,Gxn,C,mu,alpha,lspar);
    for n = 1:N
        U{n}(:) = U{n}(:) + d(sum(In(1:n-1))*R+1  : sum(In(1:n))*R);
        U{n} = max(eps,U{n});
    end
    
    % compute CP gradient with respect to U{ns}
    if isa(X,'ktensor') || isa(X,'ttensor')
        Gxnnew = mttkrp(X,U,ns);   % compute CP gradient w.r.t U{ns}
    elseif isa(X,'tensor') ||  isa(X,'sptensor')
        [Gxnnew,Pmat] = cp_gradient(U,ns,X);% compute CP gradient w.r.t U{ns}
    end
    
    P = ktensor(U);
    % Approximation error
    errf=normX2 + norm(P).^2 - 2 * sum(Gxnnew(:).*U{ns}(:));
    %errf = normX2 + norm(P)^2 - 2 * innerprod(P,X) ;
    
    % Cost function
    err2 = 1/2*errf ...
        -alpha(1:N)*cellfun(@(x) sum(log(x(:))),U) ...
        + lspar * cellfun(@(x) sum(x(:)) , U);
    
    % Rho to compute the damp parameter
    rho=real((err-err2)/(d(:)'*(Rr+mu*d(:))));                   % Eq (5.8)
    
    % Fit 
    fit = 1-(errf)/normX2; %fraction explained by model
    fitchange = abs(fit-fitold); %abs(err2 - err);
    
    negflag = min(cellfun(@(x) min(x(:)) , U)) < 0;
    if err2>err || negflag                %%% step is not accepted
        mu=mu*nu; nu=2*nu;                                      % Eq. (5.7)
        U = U1;
    else
        iter = iter+1;
        nu=2;
        mu=mu*max([1/3 1-(2*rho-1)^3]);                         % Eq. (5.7)
         
        % Normalization factors U
        am = zeros(N,R); %lambda = ones(1,R);
        for n=1:N
            am(n,:) = sum(U{n});
            %am(n,:) = max(U{n});
            %am(n,:) =sqrt(sum(U{n}.^2));
            U{n}=bsxfun(@rdivide,U{n},am(n,:));
        end
        lambdans = prod(am([1:ns-1 ns+1:end],:),1);
        lambda = lambdans .* am(ns,:);
        for n=1:N
            U{n}=bsxfun(@times,U{n},lambda.^(1/N));
        end
        
        % Precompute Cn and Gamma_n
        for n = 1:N
            C(:,:,n) = U{n}'*(U{n});
        end
        for n = 1:N
            Gamman(:,:,n) = prod(C(:,:,[1:n-1 n+1:N]),3);
        end
        
        % Fix CP gradient Gns due to normalization
        Gxn{ns} = bsxfun(@times,Gxnnew,lambda.^((N-1)/N)./lambdans);
        if ~isa(X,'ktensor') && ~isa(X,'ttensor')
            Pmat = bsxfun(@times,Pmat,lambda.^((N-ns)/N)./prod(am([ns+1:end],:),1));
        end
        % Pre-computing CP gradients
        for n = [ns-1:-1:1 ns+1:N]
            if isa(X,'ktensor') || isa(X,'ttensor') || (N<=2)
                Gxn{n} = mttkrp(X,U,n);
            elseif isa(X,'tensor') || isa(X,'sptensor')
                if (n == ns) || (n == ns+1)
                    [Gxn{n},Pmat] = cp_gradient(U,n,X);
                else
                    [Gxn{n},Pmat] = cp_gradient(U,n,Pmat);
                end
            end
        end
        
         
        fitold = fit;fitarr = [fitarr; iter fit];
        err = err2;
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n',...
                iter, fit, fitchange);
        end
    end
    if rem(iter2,3)==0
        alpha=alpha*dalpha;   %%% decrease value of alpha
    end
    if (fit>= param.fitmax) ||  ((iter > 1) && (fitchange < param.tol)) % Check for convergence
        break;
    end
end

%%
P = ktensor(U);P = arrange(P);
if param.printitn>0
    fprintf(' Final fit = %e \n', fit);
end

% Rearrange dimension of the estimation tensor 
if ~isempty(p_perm)
    P = ipermute(P,p_perm);
end

if nargout >=2
    output = struct('Uinit',{Uinit},'Fit',fitarr,'mu',mu,'nu',nu);
end

   

%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient(A,n,Pmat)
        persistent KRP_right0;
        right = N:-1:n+1; left = n-1:-1:1;
        % KRP_right =[]; KRP_left = [];
        if n <= ns
            if n == ns
                if numel(right) == 1
                    KRP_right = A{right};
                elseif numel(right) > 2
                    [KRP_right,KRP_right0] = khatrirao(A(right));
                elseif numel(right) > 1
                    KRP_right = khatrirao(A(right));
                else
                    KRP_right = 1;
                end
                
                if isa(Pmat,'tensor')
                    Pmat = reshape(Pmat.data,[],prod(In(right))); % Right-side projection
                elseif isa(Pmat,'sptensor')
                    Pmat = reshape(Pmat,[prod(size(Pmat))/prod(In(right)),prod(In(right))]); % Right-side projection
                    Pmat = spmatrix(Pmat);
                else
                    Pmat = reshape(Pmat,[],prod(In(right))); % Right-side projection
                end
                Pmat = Pmat * KRP_right ;
            else
                Pmat = reshape(Pmat,[],In(right(end)),R);
                if R>1
                    Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],In(right(end)),R));
                    Pmat = sum(Pmat,2);    % fast Right-side projection
                else
                    Pmat = Pmat * A{right(end)};
                end
            end
            
            if ~isempty(left)       % Left-side projection
                KRP_left = khatrirao(A(left));
%                 if (isempty(KRP_2) && (numel(left) > 2))
%                     [KRP_left,KRP_2] = khatrirao(A(left));
%                 elseif isempty(KRP_2)
%                     KRP_left = khatrirao(A(left));
%                     %KRP_2 = [];
%                 else
%                     KRP_left = KRP_2; KRP_2 = [];
%                 end
                T = reshape(Pmat,prod(In(left)),In(n),[]);
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
                    T = sum(T,1);
                    %G = squeeze(T);
                    G = reshape(T,[],R);
                else
                    G = (KRP_left.'*T).';
                end
            else
                %G = squeeze(Pmat);
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}.';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
                else 
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(In(left)),[]);
                elseif isa(Pmat,'sptensor')
                    T = reshape(Pmat,[prod(In(left)) prod(size(Pmat))/prod(In(left))]); % Right-side projection
                    T = spmatrix(T);
                else
                    T = reshape(Pmat,prod(In(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,In(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}.');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,In(left(1)),[]);
                    Pmat = A{left(1)}.'* Pmat;
                end
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],In(n),prod(In(right)));
                
                if (n == (ns+1)) && (numel(right)>=2)
                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[]).';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(In(right))) * KRP_right0;
                    end
                else
                    KRP_right = khatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right.',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[])';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,In(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[]).';
            end
            
        end
        
        %         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
        %             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
        %             size(KRP_left,1),size(KRP_left,2),...
        %             size(KRP_right,1),size(KRP_right,2))
    end
end


%%

function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
% defoptions = struct('tol',1e-4,'maxiters',500,'Rorder',1,...
%     'init','random','orthoforce',1,'fitmax',1,...
%     'verbose',1,'lsparse',zeros(N,1),'lcorr',zeros(N,1),'printitn',1);


param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'orth' 'fibe' 'dtld'})));
param.addOptional('alsinit',1);
param.addOptional('maxiters',500);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('tau',1e-3);
param.addOptional('maxboost',5);
param.addOptional('fullupdate',false);
param.addParamValue('MaxRecursivelevel',2);
param.addParamValue('recursivelevel',1);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);
param.addParamValue('sparse',0);

param.addParamValue('updaterule',2);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
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
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || isa(initk,'ktensor') || ... 
                    (ischar(initk)  && ismember(initk(1:4),{'fibe' 'rand' 'nvec'}))  % multi-initialization
                
                if ischar(initk)
                    cprintf('blue','Init. %d - %s',ki,initk)
                else
                    cprintf('blue','Init. %d - %s',ki,class(initk))
                end
                
                cp_fun = str2func(mfilename);
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                P = cp_fun(X,R,initparam);
                fitinit = 1- sqrt(param.normX2 + norm(P)^2 - 2 * innerprod(X,P))/normX;
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


% %% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function [d G beta alpha]= LMp_stepsize(X,U,mu,alpha,beta)
% % This function computes step size d to update v = v + d
% % d = v_ALS - vt_ALS - L Phi1\w
% % Inverse of the large Hessian is replaced by inverse of a smaller matrix
% % (NR^2 x NR^2)
% 
% N = ndims(X);In = size(X);
% R = size(U{1},2);
% UtU = zeros(R,R,N);
% for n = 1:N
%     UtU(:,:,n) = U{n}'*(U{n});
% end
% cIn = cumsum(In);
% cIn = [0 cIn];
% R2 = R^2;
% 
% Prr = per_vectrans(R,R);
% 
% ZiGZ = spalloc(N*R2,N*R2,N*R2*R2);
% iK = spalloc(N*R2,N*R2,N^2*R2);
% % K = spalloc(N*R2,N*R2,N^2*R2);
% G = zeros(sum(In)*R,1); % gradient
% iCn = zeros(R,R,sum(In));
% iCg = zeros(R*sum(In),1);
% ZiCg = zeros(R2*N,1);
% for n = 1:N
%     % Kernel matrix
% %     for m = setdiff(1:N,n)
% %         gamma = prod(UtU(:,:,setdiff(1:N,[n,m])),3);
% %         K((n-1)*R^2+1:(n)*R^2,(m-1)*R^2+1:(m)*R^2) = Prr*diag(gamma(:));
% %     end
%     
%     for m = [1:n-1 n+1:N]
%         gamma = prod(UtU(:,:,setdiff(1:N,[n,m])),3);% Gamma(n,m) Eq. (4.16)
%         iK((n-1)*R2+1:(n)*R2,(m-1)*R2+1:m*R2) = ...
%             bsxfun(@rdivide,Prr,gamma(:))*1/(N-1);   % iK(n,m),  Eq. (C.3)
%     end
%     
%     if m~=n
%         gamma = gamma .* UtU(:,:,m);                           % Gamma(n,n)
%     end
%     
%     v= UtU(:,:,n)./gamma;
%     iK((n-1)*R2+1:(n)*R2,(n-1)*R2+1:n*R2) = ...
%         -(N-2)/(N-1)*bsxfun(@times,Prr,v(:));          % iK(n,n), Eq. (C.3)
%     
%     Z = mttkrp(X,U,n);
%     Z1 = Z - U{n} * gamma;
% %     if n<=N
% %         veps = 0;
% %         reg = U{n}(:) > veps;
% %         A = [ones(sum(reg),1) -U{n}(reg)];
% %         b = -U{n}(reg) .* Z1(reg);
% %         [QQ,As] = qr(A,0);
% %         bs = QQ'*b;
% %         lb = veps *ones(2,1);
% %         x0(1) = max(veps,min(mean(b),min(b)));
% %         
% %         Z2 = Z1 + x0(1)./U{n} ;         % Eq. (4.19)
% %         reg = (Z2(:)~= 0) & (U{n}(:) > veps);
% %         betan = sum(Z2(reg))./sum(U{n}(reg));
% %         x0(2) = max(veps,min(betan,min(Z2(reg)./U{n}(reg))));
% % %         x = x0;
% % %         fun = @(x) sum(abs(A*x -b));
% % %         x = fmincon(fun,x0(:),A,b,[],[],lb,[]);
% % %         f0 = fun(x0(:));f = fun(x(:));
% %         lsopt = optimset('Display','off');
% %         [x,resnorm,residual,exitflag] = lsqlin(As,bs,A,b,[ ],[ ],lb,[],x0,lsopt);%x = lsqlin(As,bs,A,b,[ ],[ ],lb,[],x0,lsopt);
% %         f0 = sum((A*x0(:) - b).^2);
% %         f = sum((A*x(:) - b).^2);
% %         if f > f0, x = x0;end
% % %         x = A\b;
% %         %x = max(eps,x);
% %         [alpha(n), beta(n)] = deal(x(1), x(2));
% %     else
%         reg = U{n}(:) > eps;
%         alphan = max(eps,mean(-Z1(reg).*U{n}(reg)));
%         alpha(n) = max(eps,min(alphan,min(-Z1(reg).*U{n}(reg))));
%         beta(n)= 0;
% %     end
%     Unp = Z - U{n} * gamma.' + alpha(n)./U{n} - beta(n);         % Eq. (4.19)
% 
%     G(cIn(n)*R+1:cIn(n+1)*R) =  Unp(:);                          % gradient    
%     for in = 1:In(n)
%         iCn(:,:,cIn(n)+in) = inv(gamma+diag(alpha(n)./U{n}(in,:).^2 + mu));                         % Eq. (4.37)
%     end
%     
%     
%     ZiGZn = zeros(R^2,R^2);
%     for r = 1: R
%         for s = r:R
%             b = prod(U{n}(:,[r s]),2);
%             Grs = sum(bsxfun(@times,iCn(:,:,cIn(n)+1:cIn(n+1)),reshape(b,1,1,[])),3);
%             %
%             ZiGZn(R*(r-1)+1:R*r,R*(s-1)+1:R*s) = Grs;
%             if s>r
%                 ZiGZn(R*(s-1)+1:R*s,R*(r-1)+1:R*r) = Grs;
%             end
%         end
%     end
%     ZiGZ(R2*(n-1)+1:R2*n,R2*(n-1)+1:R2*n) =  ZiGZn;            % Eq. (4.39)
%     
%     % inv(C) * g
%     for in = 1:In(n)
%         iCni =  iCn(:,:,cIn(n)+in) * Unp(in,:)';
%         iCg(R*(cIn(n)+in-1)+1:R*(cIn(n)+in)) = iCni;
%     end
%     
%     % Z' * inv(C) * g
%     D = reshape(iCg(cIn(n)*R+1:cIn(n+1)*R),R,In(n)) * U{n};
%     ZiCg(R2*(n-1)+1:R2*n) = D(:);
% end
% 
% Phi = iK + ZiGZ;                                              % Eq.(4.46)
% % Phi = inv(K) + ZiGZ;                                       % Phi2 in Eq. (4.46)
% w = Phi\ZiCg;
% 
% w = reshape(w,R,R,[]);
% 
% psi = zeros(cIn(end)*R,1);
% for n = 1: N
%     Psi =  w(:,:,n) * U{n}';                     % Eq. (5.4)
%     for in = 1: In(n)
%         psi((cIn(n)+in-1)*R+1:(cIn(n)+in)*R) = iCn(:,:,cIn(n)+in) * Psi(:,in);
%     end
% end
% 
% d = iCg - psi;                                    % v_ALS - vt_ALS - L Phi\w
% 
% for n = 1:N
%     Prin = per_vectrans(R,In(n));
%     d(sum(In(1:n-1))*R+1:sum(In(1:n))*R) = Prin*d(sum(In(1:n-1))*R+1:sum(In(1:n))*R);
% end
% end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [d G beta alpha]= LMp_stepsize(X,U,Gxn,C,mu,alpha,beta)
% This function computes step size d to update v = v + d
% d = v_ALS - vt_ALS - L Phi1\w
% Inverse of the large Hessian is replaced by inverse of a smaller matrix
% (NR^2 x NR^2)
% Cn = Un'*Un;
N = ndims(X);In = size(X);
R = size(U{1},2);
% UtU = zeros(R,R,N);
% for n = 1:N
%     UtU(:,:,n) = U{n}'*(U{n});
% end
cIn = cumsum(In);
cIn = [0 cIn];
R2 = R^2;

Prr = per_vectrans(R,R);

%ZiGZ = spalloc(N*R2,N*R2,N*R2*R2);
% iK = spalloc(N*R2,N*R2,N^2*R2);
% K = spalloc(N*R2,N*R2,N^2*R2);
iK = zeros(N*R^2,N*R^2);
ZiGZ= zeros(N*R^2,N*R^2);
G = zeros(sum(In)*R,1); % gradient
iCn = zeros(R,R,sum(In));
iCg = zeros(R*sum(In),1);
ZiCg = zeros(R2*N,1);
for n = 1:N
    % Inverse of Kernel matrix
    for m = [1:n-1 n+1:N]
        gamma = prod(C(:,:,setdiff(1:N,[n,m])),3);% Gamma(n,m) Eq. (4.16)
        iK((n-1)*R2+1:(n)*R2,(m-1)*R2+1:m*R2) = ...
            bsxfun(@rdivide,Prr,gamma(:))*1/(N-1);   % iK(n,m),  Eq. (C.3)
    end
    
    if m~=n
        gamma = gamma .* C(:,:,m);                           % Gamma(n,n)
    end
    
    v= C(:,:,n)./gamma;
    iK((n-1)*R2+1:(n)*R2,(n-1)*R2+1:n*R2) = ...
        -(N-2)/(N-1)*bsxfun(@times,Prr,v(:));          % iK(n,n), Eq. (C.3)
    
    %Z = mttkrp(X,U,n);
    Z1 = Gxn{n} - U{n} * gamma;
%     if n<=N
%         veps = 0;
%         reg = U{n}(:) > veps;
%         A = [ones(sum(reg),1) -U{n}(reg)];
%         b = -U{n}(reg) .* Z1(reg);
%         [QQ,As] = qr(A,0);
%         bs = QQ'*b;
%         lb = veps *ones(2,1);
%         x0(1) = max(veps,min(mean(b),min(b)));
%         
%         Z2 = Z1 + x0(1)./U{n} ;         % Eq. (4.19)
%         reg = (Z2(:)~= 0) & (U{n}(:) > veps);
%         betan = sum(Z2(reg))./sum(U{n}(reg));
%         x0(2) = max(veps,min(betan,min(Z2(reg)./U{n}(reg))));
% %         x = x0;
% %         fun = @(x) sum(abs(A*x -b));
% %         x = fmincon(fun,x0(:),A,b,[],[],lb,[]);
% %         f0 = fun(x0(:));f = fun(x(:));
%         lsopt = optimset('Display','off');
%         [x,resnorm,residual,exitflag] = lsqlin(As,bs,A,b,[ ],[ ],lb,[],x0,lsopt);%x = lsqlin(As,bs,A,b,[ ],[ ],lb,[],x0,lsopt);
%         f0 = sum((A*x0(:) - b).^2);
%         f = sum((A*x(:) - b).^2);
%         if f > f0, x = x0;end
% %         x = A\b;
%         %x = max(eps,x);
%         [alpha(n), beta(n)] = deal(x(1), x(2));
%     else

        % Find log-barrier parameters alpha_n
        reg = U{n}(:) > eps;
        alphan = max(eps,mean(-Z1(reg).*U{n}(reg)));
        alpha(n) = max(eps,min(alphan,min(-Z1(reg).*U{n}(reg))));
        beta(n)= 0; % switch off sparse factors.
%     end
    Unp = Gxn{n} - U{n} * gamma.' + alpha(n)./U{n} - beta(n);         % Eq. (4.19)

    G(cIn(n)*R+1:cIn(n+1)*R) =  Unp(:);                          % gradient    
    for in = 1:In(n)
        iCn(:,:,cIn(n)+in) = inv(gamma+diag(alpha(n)./U{n}(in,:).^2 + mu));                         % Eq. (4.37)
    end
    
    
    ZiGZn = zeros(R^2,R^2);
    for r = 1: R
        for s = r:R
            b = prod(U{n}(:,[r s]),2);
            Grs = sum(bsxfun(@times,iCn(:,:,cIn(n)+1:cIn(n+1)),reshape(b,1,1,[])),3);
            %
            ZiGZn(R*(r-1)+1:R*r,R*(s-1)+1:R*s) = Grs;
            if s>r
                ZiGZn(R*(s-1)+1:R*s,R*(r-1)+1:R*r) = Grs;
            end
        end
    end
    ZiGZ(R2*(n-1)+1:R2*n,R2*(n-1)+1:R2*n) =  ZiGZn;            % Eq. (4.39)
    
    % inv(C) * g
    for in = 1:In(n)
        iCni =  iCn(:,:,cIn(n)+in) * Unp(in,:)';
        iCg(R*(cIn(n)+in-1)+1:R*(cIn(n)+in)) = iCni;
    end
    
    % Z' * inv(C) * g
    D = reshape(iCg(cIn(n)*R+1:cIn(n+1)*R),R,In(n)) * U{n};
    ZiCg(R2*(n-1)+1:R2*n) = D(:);
end

Phi = iK + ZiGZ;                                              % Eq.(4.46)
% Phi = inv(K) + ZiGZ;                                       % Phi2 in Eq. (4.46)
w = Phi\ZiCg;

w = reshape(w,R,R,[]);

psi = zeros(cIn(end)*R,1);
for n = 1: N
    Psi =  w(:,:,n) * U{n}';                     % Eq. (5.4)
    for in = 1: In(n)
        psi((cIn(n)+in-1)*R+1:(cIn(n)+in)*R) = iCn(:,:,cIn(n)+in) * Psi(:,in);
    end
end

d = iCg - psi;                                    % v_ALS - vt_ALS - L Phi\w

for n = 1:N
    Prin = per_vectrans(R,In(n));
    d(sum(In(1:n-1))*R+1:sum(In(1:n))*R) = Prin*d(sum(In(1:n-1))*R+1:sum(In(1:n))*R);
end
end

function P = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
%  vectorize of transposition
M = reshape(1:m*n,[],n);
Mt = M';Perm = Mt(:);
P = speye(m*n);%eye(numel(M));
P = P(Perm,:);
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = khatrirao_t(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1}';

for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i}',R,[]),reshape(K,R,1,[]));
end
K = reshape(K,R,[]);

end


%%
function K = kron(A,B)
%  Fast implementation of Kronecker product of A and B
%
%   Copyright 2012 Phan Anh Huy
%   $Date: 2012/3/18$

if ndims(A) > 2 || ndims(B) > 2
    error(message('See ndkron.m'));
end
I = size(A); J = size(B);

if ~issparse(A) && ~issparse(B)
    K = bsxfun(@times,reshape(B,J(1),1,J(2),1),reshape(A,1,I(1),1,I(2)));
    K = reshape(K,I(1)*J(1),[]);
else
    K = kron(A,B);
end
end