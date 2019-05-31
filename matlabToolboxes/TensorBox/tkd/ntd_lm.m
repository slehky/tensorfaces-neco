function [T,output] = ntd_lm(Y,R,opts)
% LM - Damped Gauss-Newton (Levenberg-Marquard) algorithm with log-barrier
% function factorizes an N-way tensor X into factors and a core tensor with
%       nonnegativity constraints.
% 
% The algorithm estimates all factor matrices and the core tensor
% simultaneously. 
% For decompositions with relatively large factor matrices or core tensors,
% e.g., the number of parameters exceeds 10.000, simplified versions of the
% algorithms are suggested to use.
%    - O2LB:  an alternating algorithm estimates only one factor matrix or
%    core tensor at a time.
%    - LMPU:  an algorithm estimates a factor matrix and a core tensor
%    at a time, instead of all factor matrices and a core tensor. 
%
% INPUT:
%   X:  N-way data which can be a tensor or a ttensor.
%   R:  multilinear rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'nmfs'|'fiber'| ttensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'nmfs' 'nvec'};
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%
% OUTPUT: 
%  T:  ttensor of estimated factors and core tensor
%  output:  
%      .Fit
%      .NoIters 
%
% EXAMPLE
%   X = tensor(rand([10 20 30]));  
%   opts = ntd_lm;
%   opts.init = {'nvec' 'nmfs' 'random'};
%   [P,output] = ntd_lm(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
% 
% [1] Anh-Huy Phan; Tichavsky, P.; Cichocki, A., "Damped Gauss-Newton
% algorithm for nonnegative Tucker decomposition," Statistical Signal
% Processing Workshop (SSP), 2011 IEEE , vol., no., pp.665,668, 28-30 June
% 2011.
%
% [2] Anh-Huy Phan, "Algorithms for Tensor Decompositions and
% Applications", PhD thesis, Kyutech, Japan, 2011.
% 
% The function uses the Matlab Tensor toolbox.
% See also: tucker_als, ntd_o2lb, ntd_lmpu
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors.
%
% This algorithm is a part of the TENSORBOX, 2012.


if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    T = param; return
end

N = ndims(Y); In = size(Y);
if isempty(param.normX)
    normY = norm(Y);
else
    normY = param.normY;
end

if numel(R) == 1
    R = R(ones(1,N));
end

%% Initialization
initopts = ntd_o2lb;
initopts.alsinit = 0;
initopts.init = param.init;
initopts.alsinit = param.alsinit;
initopts.maxiters = 10;

initopts.ntd_func = @ntd_o2lb;

Yhat = initopts.ntd_func(Y,R,initopts);
A = Yhat.u(:); G = tensor(Yhat.core);

if nargout >=2
    output = struct('init',{[A(:) ; {G}]},'Fit',[],'mu',[],'nu',[]);
end

% p_perm = [];
% if ~issorted(In)
%     [In,p_perm] = sort(In);
%     Y = permute(Y,p_perm);
%     G = permute(G,p_perm);
%     A = A(p_perm);
% end
%%
fprintf('\nfLM for NTD:\n');

%%
nu=2; tau=1;
% ell2 = zeros(N,R);
% for n = 1:N
%     ell2(n,:) = sum(abs(U{n}).^2);
% end
% mm = zeros(N,R);
% for n = 1:N
%     mm(n,:) = prod(ell2([1:n-1 n+1:N],:),1);
% end
% mu=tau*max(mm(:));
mu = (max(G(:)))^2;
alpha = mean(G(:).^2);
% mu = 1e4;
%alpha = 10;   %%% initial multiplicative constant for the barrier function
%dalpha=exp(100/opts.maxiters*log(0.005));
% dalpha = 0.001;

warning off;
%% Main Loop: Iterate until convergence
% iter2 = 1;boostcnt = 0;
flagtol = 0;
% mu_inc_count = 0;

[err2,errest] = costvalue(Y,G,A,alpha,normY);
fit = 1-err2/normY;
fitarr = [];

for iter = 1:param.maxiters
    pause(0.001)
    fitold = fit;
    
   
    [Hreg,gdreg,g,alpha] = gradHess(Y,G,A);
    
    Aold = A;Gold = G;
    err = err2;
    for iter2 = 1:10
        Hreg2 = Hreg;
        dh = Hreg2(1:size(Hreg2,1)+1:end);
        Hreg2(1:size(Hreg2,1)+1:end) = dh + mu;
        d = Hreg2\gdreg;
        
        for n = 1:N
            dd = d(In(1:n-1)*R(1:n-1)'+1  : In(1:n)*R(1:n)');
            A{n} = Aold{n} + reshape(dd,R(n),[])';
            A{n} = max(eps,A{n});
        end
        G(:) = Gold(:) + d(In * R'+1:end);
        G(:) = max(eps,G(:));


        [err2,err0] = costvalue(Y,G,A,alpha,normY);
        rho=real((err-err2)/(d(:)'*(g+mu*d(:))));
        
        if err2>err
            mu = mu*nu; nu=2*nu;
        else
            nu=2;
            mu=mu*max([1/3 1-(2*rho-1)^3]);
            for n=1:N
                am=sqrt(sum(A{n}.^2));          % normalization
                A{n}=bsxfun(@rdivide,A{n},am);
                G = ttm(G,diag(am),n);
            end
            
            errest = err0;
            break;
        end
    end
        
    fit = 1-sqrt(abs(errest))/normY; %fraction explained by model
    fitchange = abs(fitold - fit);
    
    if mod(iter,param.printitn)==0
        fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n',...
            iter, fit, fitchange);
    end
    
    fitarr = [fitarr; iter  fit];
    % Check for convergence
    if (iter > 5) && ((fitchange < param.tol) || (fit >= param.fitmax) ) % Check for convergence
        flagtol = flagtol + 1;
    else
        flagtol = 0;
    end
    
    if flagtol >= 5,
        break
    end
    
    %     if rem(iter,5)==0
    %         alpha=alpha*dalpha;   %%% decrease value of alpha
    %     end
    %
end


% for iter = 1:param.maxiters
%     pause(0.001)
%     fitold = fit;
%     err = err2;
%     
%     A1 = A;G1 = G;
%     
%     [G,A,d, Rr,alpha] = lmupdate(Y,G,A,mu,alpha);    
%     
%     [err2,err0] = costvalue(Y,G,A,alpha,normY);
%     rho=real((err-err2)/(d(:)'*(Rr+mu*d(:))));
%     
%     if err2>err
%         mu=mu*nu; nu=2*nu;
%         A = A1;G = G1;
%         err2 = err;
%     else
%         nu=2;
%         mu=mu*max([1/3 1-(2*rho-1)^3]);
%         for n=1:N
%             am=sqrt(sum(A{n}.^2));          % normalization
%             A{n}=bsxfun(@rdivide,A{n},am);
%             G = ttm(G,diag(am),n);
%         end
%         
%         fit = 1-sqrt(abs(err0))/normY; %fraction explained by model
%         fitchange = abs(fitold - fit);
%         fitarr = [fitarr; iter  fit];
%         if mod(iter,param.printitn)==0
%             fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n',...
%                 iter, fit, fitchange);
%         end
%         
%     end
%     % Check for convergence after at least 5 iterations
%     if (iter > 5) && ((fitchange < param.tol) || (fit >= param.fitmax) ) % Check for convergence
%         flagtol = flagtol + 1;
%     else
%         flagtol = 0;
%     end
%     
%     if flagtol >= 5, % stop when there are 5 consecutive approximation errors lower than tol.
%         break
%     end
%     
%     %     if rem(iter,5)==0
%     %         alpha=alpha*dalpha;   %%% decrease value of alpha
%     %     end
%     %
% end

%% Compute the final result
T = ttensor(G, A);
% % Rearrange dimension of the estimation tensor 
% if ~isempty(p_perm)
%     T = ipermute(T,p_perm);
% end

if nargout >=2
    output.Fit = fitarr;
    output.mu  = mu;
    output.nu = nu;
end

end



%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','nmfs',@(x) (iscell(x) || isa(x,'ttensor')||ismember(x(1:4),{'rand' 'nvec' 'fibe' 'nmfs'})));
param.addOptional('alsinit',1);
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
% param.addOptional('linesearch',true);
% param.addParamValue('TraceFit',false,@islogical);
% param.addParamValue('TraceMSAE',true,@islogical);

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;
end

%%
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Hreg,gdreg2,g,alpha]= gradHess(X,G,A)
% return gradient and Hessian of the function 
% f(G,A) = \| X - G x {A}\|_F^2 - sum_n alpha_n sum(log(A{n})) - alpha_G sum(log(G));
%
persistent Prnin H Inc Rc gd Pnrind_nm Pnmind Pqind2 Pc;

N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);R = R(:);
cI = In(:).*R(:);
cI = cumsum([0 cI']);
pR = prod(R);

if isempty(Prnin) || (numel(Prnin) ~= N) || ~isequal(R,Rc) || ~isequal(Inc,In)
    H = zeros(cI(end)+pR);
    Rc = R; Inc = In;
    gd = zeros(cI(end)+pR,1);

    Pqind2= cell(N,1);
    Pc = cell(N,N);
    for n = 1:N
        Prnin{n} = per_vectranst(In(n),R(n));
        for m = n+1:N
            Pnmind{n,m} = per_vectranst(In(n),In(m));
            Pnrind_nm{n,m} = permute_vec(R([1:n n:m-1 m+1:end])',n);
            
            Iy = [R(1:n); R(n:m-1) ;In(m); R(m+1:end)]';   % R1 x... x R(n-1) x R(n) x R(n) x ... R(m-1) x I_m x R(m+1) x ... x R_N
            
            Pqind = per_vectranst(pR/R(m)*In(m),R(n));
            Pyp1ind_nm_1 = permute_vec(Iy,n+1);
            Pyp1ind_nm_2  = permute_vec(Iy,m+1);
            
            ind = (1:size(Pqind,1))';
            ind = ind(Pqind,:); ind(Pyp1ind_nm_1) = ind; ind = ind(Pyp1ind_nm_2,:);

            Pc{n,m} = ind;
        end
        
        Iy = R([1:n n:end])';
        Pynp1ind = permute_vec(Iy,n+1);
        Pymp1ind = permute_vec(Iy,n);
        Pq2ind = per_vectranst(pR,R(n));
        Pq2ind(Pynp1ind) = Pq2ind; Pq2ind = Pq2ind(Pymp1ind);
        
        Pqind2{n} = Pq2ind;
    end
end

%% Full Hessian
AtA = cellfun(@(x) x'*x , A,'uni',0);
Gmat = cell(N,1);
for n = 1:N
    Gmat{n} = double(tenmat(G,n));
end

for n = 1:N
    % Jn = grad(y)/grad(vec(An^T))
    %%%  Jn' Jm
    for m = n+1:N
        %  Fast way without computing Jacobian, avoid Kronkect products
        % Phi_n * B (kronecker products of AtA...)
        GtAA = ttm(G,AtA,-[n m]);
        W = ttm(GtAA,A{m},m);
        Y = W(:) * reshape(A{n}',[],1)';
        Y = reshape(Y,[],In(n));
        Y = Y(Pc{n,m},:);
 
        Y = reshape(Y',In(m)*In(n),[]);
        Y = Y(Pnmind{n,m},Pnrind_nm{n,m});        %Y = Pnm * Y * Pnr';
        
        Y = reshape(Y,In(m),In(n),R(n),[]);
        Y = permute(Y,[3 1 2 4]);
        Y = reshape(Y,In(m)*In(n)*R(n),[]);
        
        Prnrmind = per_vectranst(R(n),R(m));
        Hinm = Gmat{m} * Y';
        Hinm = reshape(Hinm,R(n) * R(m), []);
        Hinm(Prnrmind,:) = Hinm;
        Hnm2r = reshape(Hinm,R(n),[],In(n));
        Hnm2r = permute(Hnm2r,[1 3 2]);
        Hnm2r = reshape(Hnm2r,R(n) * In(n),[]);
        
        %% Hnm and Hmn
        H(cI(n)+1:cI(n+1),cI(m)+1:cI(m+1)) = Hnm2r;% size (R(n) * In(n)) x (R(n) * In(n))
        H(cI(m)+1:cI(m+1),cI(n)+1:cI(n+1)) = Hnm2r';
    end
    
    %%  Jn' Jn
    if n < N
        GtAA = ttm(GtAA,AtA{N},N);
    else
        GtAA = ttm(G,AtA,-n);
    end
    GtAAn = double(tenmat(GtAA,n));
    TT = GtAAn * Gmat{n}';
 
    for ii = 1:In(n)
        H(cI(n)+1+(ii-1)*R(n):cI(n)+ii*R(n),cI(n)+1+(ii-1)*R(n):cI(n)+ii*R(n)) = TT; %Hnn
    end
    
    %%  Jn' JN+1 
    Y = GtAA(:) * A{n}(Prnin{n})';
    Y = reshape(Y,[],In(n))';
    Y = Y(:,Pqind2{n});%Y = Y * Pq;
    HnN = reshape(Y,R(n)*In(n),[]);
    HnN = HnN(Prnin{n},:);
    
%     Y = reshape(A{n}',[],1) * GtAA(:)';
%     Y = reshape(Y,[R(n) In(n)*prod(R(1:n-1)) R(n) prod(R(n+1:end))]);
%     Y = permute(Y,[3 2 1 4]);
%     HnN = reshape(Y,R(n)*In(n),[]);
%     norm(HnN - HnN2,'fro')
    %%
    H(cI(n)+1:cI(n+1),cI(end)+1:end) = HnN;
    H(cI(end)+1:end,cI(n)+1:cI(n+1)) = HnN';
    
    %% gradient Gn = grad(f)/grad(vec(A{n}')
    Xp = ttm(X,A,-n,'t');Xp = tenmat(Xp,n);
    gn = Xp.data * Gmat{n}.' - A{n}*TT; gn = gn';
    gd(cI(n)+1:cI(n+1)) = gn(:);
end

Hnn = AtA{N};
for n = N-1:-1:1
    Hnn = kron(Hnn,AtA{n});
end
H(cI(end)+1:end,cI(end)+1:end) = Hnn;
 
% Gradient grad(f)/grad(vec(G)
gd(cI(end)+1:end) = reshape(Xp.data.'*A{N},[],1) - reshape(GtAAn.'*AtA{N},[],1);
g = gd;
%%
v = cellfun(@(x) reshape(x',[],1), A,'uni',0);
v = cell2mat(v(:));
v = [v; G(:)] + eps;

%%
ivn= v;
alphan = nan(N+1,1);
for n = 1:N
    vn = v(cI(n)+1:cI(n+1));
    gdn = gd(cI(n)+1:cI(n+1));
    msk = (vn~=0) & (gdn~= 0);
    alphan(n) = max([eps,min(-gdn(msk).*vn(msk))]);
    ivn(cI(n)+1:cI(n+1)) = alphan(n)./vn;
end
vn = v(end-pR+1:end);
gdn = gd(end-pR+1:end);
msk = (vn~=0) & (gdn~= 0);
alphan(N+1) = max([eps,min(-gdn(msk).*vn(msk))]);
ivn(end-pR+1:end) = alphan(N+1)./vn;

gdreg2 =  gd + ivn;
Hreg = H;
dH = H(1:size(H,1)+1:end) ;
Hreg(1:size(H,1)+1:end) = dH(:) + ivn./v;

% d = Hreg\gdreg2;

% % [G2,A2] = update(G,A,d,R,In,N);
% In = In(:)';R = R(:)';
% for n = 1:N
%     dd = d(In(1:n-1)*R(1:n-1)'+1  : In(1:n)*R(1:n)');
%     A{n} = A{n} + reshape(dd,R(n),[])';
%     A{n} = max(eps,A{n});
% end
% G(:) = G(:) + d(In * R'+1:end);
% G(:) = max(eps,G(:));

alpha = alphan;
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [G,A,d,g,alpha]= lmupdate(X,G,A,mu,alpha)
%
persistent Prnin H Inc Rc gd Pnrind_nm Pnmind Pqind2 Pc;

N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);R = R(:);
cI = In(:).*R(:);
cI = cumsum([0 cI']);
pR = prod(R);

if isempty(Prnin) || (numel(Prnin) ~= N) || ~isequal(R,Rc) || ~isequal(Inc,In)
    H = zeros(cI(end)+pR);
    Rc = R; Inc = In;
    gd = zeros(cI(end)+pR,1);

    Pqind2= cell(N,1);
    Pc = cell(N,N);
    for n = 1:N
        Prnin{n} = per_vectranst(In(n),R(n));
        for m = n+1:N
            Pnmind{n,m} = per_vectranst(In(n),In(m));
            Pnrind_nm{n,m} = permute_vec(R([1:n n:m-1 m+1:end])',n);
            
            Iy = [R(1:n); R(n:m-1) ;In(m); R(m+1:end)]';   % R1 x... x R(n-1) x R(n) x R(n) x ... R(m-1) x I_m x R(m+1) x ... x R_N
            
            Pqind = per_vectranst(pR/R(m)*In(m),R(n));
            Pyp1ind_nm_1 = permute_vec(Iy,n+1);
            Pyp1ind_nm_2  = permute_vec(Iy,m+1);
            
            ind = (1:size(Pqind,1))';
            ind = ind(Pqind,:); ind(Pyp1ind_nm_1) = ind; ind = ind(Pyp1ind_nm_2,:);

            Pc{n,m} = ind;
        end
        
        Iy = R([1:n n:end])';
        Pynp1ind = permute_vec(Iy,n+1);
        Pymp1ind = permute_vec(Iy,n);
        Pq2ind = per_vectranst(pR,R(n));
        Pq2ind(Pynp1ind) = Pq2ind; Pq2ind = Pq2ind(Pymp1ind);
        
        Pqind2{n} = Pq2ind;
    end
end

%% Full Hessian
AtA = cellfun(@(x) x'*x , A,'uni',0);
Gmat = cell(N,1);
for n = 1:N
    Gmat{n} = double(tenmat(G,n));
end

%%

for n = 1:N
    % Jn = grad(y)/grad(vec(An^T))
    %%%  Jn' Jm
    for m = n+1:N
        %  Fast way without computing Jacobian, avoid Kronkect products
        % Phi_n * B (kronecker products of AtA...)
        GtAA = ttm(G,AtA,-[n m]);
        W = ttm(GtAA,A{m},m);
        Y = W(:) * reshape(A{n}',[],1)';
        Y = reshape(Y,[],In(n));
        Y = Y(Pc{n,m},:);
 
        Y = reshape(Y',In(m)*In(n),[]);
        Y = Y(Pnmind{n,m},Pnrind_nm{n,m});        %Y = Pnm * Y * Pnr';
        
        Y = reshape(Y,In(m),In(n),R(n),[]);
        Y = permute(Y,[3 1 2 4]);
        Y = reshape(Y,In(m)*In(n)*R(n),[]);
        
        Prnrmind = per_vectranst(R(n),R(m));
        Hinm = Gmat{m} * Y';
        Hinm = reshape(Hinm,R(n) * R(m), []);
        Hinm(Prnrmind,:) = Hinm;
        Hnm2r = reshape(Hinm,R(n),[],In(n));
        Hnm2r = permute(Hnm2r,[1 3 2]);
        Hnm2r = reshape(Hnm2r,R(n) * In(n),[]);
        
        %% Hnm and Hmn
        H(cI(n)+1:cI(n+1),cI(m)+1:cI(m+1)) = Hnm2r;% size (R(n) * In(n)) x (R(n) * In(n))
        H(cI(m)+1:cI(m+1),cI(n)+1:cI(n+1)) = Hnm2r';
    end
    
    %%  Jn' Jn
    if n < N
        GtAA = ttm(GtAA,AtA{N},N);
    else
        GtAA = ttm(G,AtA,-n);
    end
    GtAAn = double(tenmat(GtAA,n));
    TT = GtAAn * Gmat{n}';
 
    for ii = 1:In(n)
        H(cI(n)+1+(ii-1)*R(n):cI(n)+ii*R(n),cI(n)+1+(ii-1)*R(n):cI(n)+ii*R(n)) = TT; %Hnn
    end
    
    %%  Jn' JN+1 
    Y = GtAA(:) * A{n}(Prnin{n})';
    Y = reshape(Y,[],In(n))';
    Y = Y(:,Pqind2{n});%Y = Y * Pq;
    HnN = reshape(Y,R(n)*In(n),[]);
    HnN = HnN(Prnin{n},:);
    
%     Y = reshape(A{n}',[],1) * GtAA(:)';
%     Y = reshape(Y,[R(n) In(n)*prod(R(1:n-1)) R(n) prod(R(n+1:end))]);
%     Y = permute(Y,[3 2 1 4]);
%     HnN = reshape(Y,R(n)*In(n),[]);
%     norm(HnN - HnN2,'fro')
    %%
    H(cI(n)+1:cI(n+1),cI(end)+1:end) = HnN;
    H(cI(end)+1:end,cI(n)+1:cI(n+1)) = HnN';
    
    %% gradient Gn = grad(f)/grad(vec(A{n}')
    Xp = ttm(X,A,-n,'t');Xp = tenmat(Xp,n);
    gn = Xp.data * Gmat{n}.' - A{n}*TT; gn = gn';
    gd(cI(n)+1:cI(n+1)) = gn(:);
end

Hnn = AtA{N};
for n = N-1:-1:1
    Hnn = kron(Hnn,AtA{n});
end
H(cI(end)+1:end,cI(end)+1:end) = Hnn;
 
% Gradient grad(f)/grad(vec(G)
gd(cI(end)+1:end) = reshape(Xp.data.'*A{N},[],1) - reshape(GtAAn.'*AtA{N},[],1);
g = gd;
%%
v = cellfun(@(x) reshape(x',[],1), A,'uni',0);
v = cell2mat(v(:));
v = [v; G(:)];

%%
ivn= v;
alphan = nan(N+1,1);
for n = 1:N
    vn = v(cI(n)+1:cI(n+1));
    gdn = gd(cI(n)+1:cI(n+1));
    msk = (vn~=0) & (gdn~= 0);
    alphan(n) = max([eps,min(-gdn(msk).*vn(msk))]);
    ivn(cI(n)+1:cI(n+1)) = alphan(n)./vn;
end
vn = v(end-pR+1:end);
gdn = gd(end-pR+1:end);
msk = (vn~=0) & (gdn~= 0);
alphan(N+1) = max([eps,min(-gdn(msk).*vn(msk))]);
ivn(end-pR+1:end) = alphan(N+1)./vn;

gdreg2 =  gd + ivn;
Hreg = H;
dH = H(1:size(H,1)+1:end) ;
Hreg(1:size(H,1)+1:end) = dH(:) + ivn./v + mu;

d = Hreg\gdreg2;

% [G2,A2] = update(G,A,d,R,In,N);
In = In(:)';R = R(:)';
for n = 1:N
    dd = d(In(1:n-1)*R(1:n-1)'+1  : In(1:n)*R(1:n)');
    A{n} = A{n} + reshape(dd,R(n),[])';
    A{n} = max(eps,A{n});
end
G(:) = G(:) + d(In * R'+1:end);
G(:) = max(eps,G(:));

alpha = alphan;
end


function [err,err0] = costvalue(X,G,A,alpha,normX)
N = ndims(X);
XA = ttm(X,A,'t');
GAtA = full(ttm(ttensor(G,A),A,'t'));
err0 = normX^2 + (GAtA(:) - 2*XA(:)).' * G(:);
%Xhat = ttensor(G,A);
%err0=normX.^2 + norm(Xhat).^2 - 2 * innerprod(X,Xhat);
%err0 = double(err0);
% ldlog = cellfun(@(x) sum(log(x(:))), A);

%err = err0-alpha*(sum(ldlog) + sum(log(G(:))));
if numel(alpha) == 1
    alpha = alpha(ones(N+1),1);
end
err = err0;
for n = 1:N
    err = err - alpha(n) * sum(log(A{n}(:)));
end
err = err - alpha(n+1) * sum(log(G(:)));
end


function [Perm,P] = per_vectranst(m,n)
% vec(X_mn) = P vec(X^T)
%  vectorize of transposition
M = reshape(1:m*n,[],n);
Mt = M';Perm = Mt(:);
if nargout >1
    P = speye(m*n);%eye(numel(M));
    P(Perm,:) = P;
end
 
end


function [Perm,P] = permute_vec(In,n)
% vec(X_n) = P * vec(X)
Tt = reshape(1:prod(In),In);N = numel(In);
%Tn = tenmat(Tt,n);
Tn = permute(Tt,[n 1:n-1 n+1:N]);
Perm = Tn(:);
if nargout > 1
    P = speye(prod(In));
    P(:,Perm) = P;
end
end
