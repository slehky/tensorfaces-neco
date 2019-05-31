function [T,output] = ntd_lmpu(Y,R,opts)
% LMPU - Damped Gauss-Newton (Levenberg-Marquard) algorithm factorizes
%       an N-way tensor X into N factors and a core tensor with
%       nonnegativity constraints.
%
% The algorithm is simplified from the LM algorithm which estimates all
% factor matrices and the core tensor at a time. This algorithm estimates
% only 1 factor matrix and the core tensor.
%
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
%   R:  multilinear rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'| ttensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'random' 'nvec'};
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
%   opts = ntd_lmpu;
%   opts.init = {'nvec' 'nmfs' 'random'};
%   [P,output] = ntd_lmpu(X,5,opts);
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
% See also: ntd_lm, ntd_o2lb
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by the authors.
%
% This algorithm is a part of the TENSORBOX, 2013.


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
mu = mu(ones(1,N));
[err2,errest] = costvalue(Y,G,A,alpha,normY);

fit = 1-sqrt(abs(errest))/normY; %fraction explained by model
fprintf(' Iter %2d: fit = %e \n', 0, fit);

fitarr = [1 fit];


for iter = 1:param.maxiters
    pause(0.001)
    fitold = fit;

%     % TYPE 1
%     for n = 1:N
%         err = err2;
%         A1 = A;G1 = G;
%         
%         for iter2 = 1:10
%             [G,A,d, Rr,alpha] = lmupdate_AG(Y,G,A,mu(n),alpha,normY,n);
%             
%             [err2,err0] = costvalue(Y,G,A,alpha,normY);
%             rho=real((err-err2)/(d(:)'*(Rr+mu(n)*d(:))));
%             
%             if err2>err                                        %% The step is not accepted
%                 mu(n) = mu(n)*nu; nu=2*nu;
%                 A = A1;G = G1;
%                 err2 = err;
%             else
%                 nu=2;
%                 mu(n)=mu(n)*max([1/3 1-(2*rho-1)^3]);
%                 am=sqrt(sum(A{n}.^2));   %%% new normalization
%                 A{n}=bsxfun(@rdivide,A{n},am);
%                 G = ttm(G,diag(am),n);
%                 errest = err0;
%                 break;
%             end
%         end
%     end

    % Type 2
    for n = 1:N
        err = err2;
        [Hreg,greg,g,alpha] = gradHess(Y,G,A,alpha,n);
        Aold = A;Gold = G;
        for iter2 = 1:10
            Hreg2 = Hreg;
            dh = Hreg2(1:size(Hreg2,1)+1:end);
            Hreg2(1:size(Hreg2,1)+1:end) = dh + mu(n);
            d = Hreg2\greg;
            
            dd = d(1:In(n)*R(n)');
            A{n} = Aold{n} + reshape(dd,R(n),[])';
            A{n} = max(eps,A{n});
            G(:) = Gold(:) + d(In(n) * R(n)+1:end);
            G(:) = max(eps,G(:));
            
            [err2,err0] = costvalue(Y,G,A,alpha,normY);
            rho=real((err-err2)/(d(:)'*(g+mu(n)*d(:))));
            
            if err2>err 
                mu(n) = mu(n)*nu; nu=2*nu;
            else
                nu=2;
                mu(n)=mu(n)*max([1/3 1-(2*rho-1)^3]);
                am=sqrt(sum(A{n}.^2));   %%% new normalization
                A{n}=bsxfun(@rdivide,A{n},am);
                G = ttm(G,diag(am),n);
                errest = err0;
                break;
            end
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
    
end
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



%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Hreg,gdreg2,g,alpha]= gradHess(X,G,A,alpha,mode)
%
persistent Prnin Inc Rc  Pqind2;

N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);R = R(:);
pR = prod(R);

if isempty(Prnin) || (numel(Prnin) ~= N) || ~isequal(R,Rc) || ~isequal(Inc,In)
    Rc = R; Inc = In;

    Pqind2= cell(N,1);
    for n = 1:N
        Prnin{n} = per_vectranst(In(n),R(n));

        Iy = R([1:n n:end])';
        Pynp1ind = permute_vec(Iy,n+1);
        Pymp1ind = permute_vec(Iy,n);
        Pq2ind = per_vectranst(pR,R(n));
        Pq2ind(Pynp1ind) = Pq2ind; Pq2ind = Pq2ind(Pymp1ind);
        
        Pqind2{n} = Pq2ind;
    end
end

sz = In(mode)*R(mode)+prod(R);
gd = zeros(sz,1);
H = zeros(sz,sz);

%% Full Hessian
AtA = cellfun(@(x) x'*x , A,'uni',0);
Gmat = cell(N,1);
for n = 1:N
    Gmat{n} = double(tenmat(G,n));
end

%%
for n = mode
    % Jn = grad(y)/grad(vec(An^T))
    
    %%  Jn' Jn
    
    GtAA = ttm(G,AtA,-n);
    GtAAn = double(tenmat(GtAA,n));
    TT = GtAAn * Gmat{n}';
 
    for ii = 1:In(n)
        H(1+(ii-1)*R(n):ii*R(n),1+(ii-1)*R(n):ii*R(n)) = TT; %Hnn
    end
    
    %%  Jn' JN+1 
    Y = GtAA(:) * A{n}(Prnin{n})';
    Y = reshape(Y,[],In(n))';
    Y = Y(:,Pqind2{n});%Y = Y * Pq;
    HnN = reshape(Y,R(n)*In(n),[]);
    HnN = HnN(Prnin{n},:);
    
     
    H(1:In(n)*R(n),In(n)*R(n)+1:end) = HnN;
    H(In(n)*R(n)+1:end,1:In(n)*R(n)) = HnN';
    
    %% gradient Gn = grad(f)/grad(vec(A{n}')
    Xp = ttm(X,A,-n,'t');Xp = tenmat(Xp,n);
    gn = Xp.data * Gmat{n}.' - A{n}*TT; gn = gn';
    gd(1:In(n)*R(n)) = gn(:);
end

Hnn = AtA{N};
for n = N-1:-1:1
    Hnn = kron(Hnn,AtA{n});
end
H(In(mode)*R(mode)+1:end,In(mode)*R(mode)+1:end) = Hnn;
 
% Gradient grad(f)/grad(vec(G)
gG = A{mode}.'*Xp.data - AtA{mode}.'*GtAAn;
gG = reshape(gG,[R(mode) prod(R(1:mode-1)) prod(R(mode+1:N))]);
gG = permute(gG,[2 1 3]);
gd(In(mode)*R(mode)+1:end) = gG(:);
g = gd;

%%
v = [reshape(A{mode}',[],1); G(:)]+eps;

ivn= v;
alphan = nan(N+1,1);
for n = mode
    vn = v(1:In(n)*R(n));
    gdn = gd(1:In(n)*R(n));
    msk = (vn~=0) & (gdn~= 0);
    alphan(n) = max([eps,min(-gdn(msk).*vn(msk))]);
    ivn(1:In(n)*R(n)) = alphan(n)./vn;
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

% (Hreg + mu I )\ gdreg2

alpha([mode N+1]) = alphan([mode end]);
end

%%
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


% %% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function [G,A,d,g,alpha,Hreg]= lmupdate_AG(X,G,A,mu,alpha,normX,mode)
% %
% persistent Prnin Inc Rc  Pqind2;
% 
% N = ndims(X);In = size(X);
% R = cellfun(@(x) size(x,2),A);R = R(:);
% pR = prod(R);
% 
% if isempty(Prnin) || (numel(Prnin) ~= N) || ~isequal(R,Rc) || ~isequal(Inc,In)
%     Rc = R; Inc = In;
% 
%     Pqind2= cell(N,1);
%     for n = 1:N
%         Prnin{n} = per_vectranst(In(n),R(n));
% 
%         Iy = R([1:n n:end])';
%         Pynp1ind = permute_vec(Iy,n+1);
%         Pymp1ind = permute_vec(Iy,n);
%         Pq2ind = per_vectranst(pR,R(n));
%         Pq2ind(Pynp1ind) = Pq2ind; Pq2ind = Pq2ind(Pymp1ind);
%         
%         Pqind2{n} = Pq2ind;
%     end
% end
% 
% sz = In(mode)*R(mode)+prod(R);
% gd = zeros(sz,1);
% H = zeros(sz,sz);
% 
% %% Full Hessian
% AtA = cellfun(@(x) x'*x , A,'uni',0);
% Gmat = cell(N,1);
% for n = 1:N
%     Gmat{n} = double(tenmat(G,n));
% end
% 
% %%
% for n = mode
%     % Jn = grad(y)/grad(vec(An^T))
%     
%     %%  Jn' Jn
%     
%     GtAA = ttm(G,AtA,-n);
%     GtAAn = double(tenmat(GtAA,n));
%     TT = GtAAn * Gmat{n}';
%  
%     for ii = 1:In(n)
%         H(1+(ii-1)*R(n):ii*R(n),1+(ii-1)*R(n):ii*R(n)) = TT; %Hnn
%     end
%     
%     %%  Jn' JN+1 
%     Y = GtAA(:) * A{n}(Prnin{n})';
%     Y = reshape(Y,[],In(n))';
%     Y = Y(:,Pqind2{n});%Y = Y * Pq;
%     HnN = reshape(Y,R(n)*In(n),[]);
%     HnN = HnN(Prnin{n},:);
%     
%      
%     H(1:In(n)*R(n),In(n)*R(n)+1:end) = HnN;
%     H(In(n)*R(n)+1:end,1:In(n)*R(n)) = HnN';
%     
%     %% gradient Gn = grad(f)/grad(vec(A{n}')
%     Xp = ttm(X,A,-n,'t');Xp = tenmat(Xp,n);
%     gn = Xp.data * Gmat{n}.' - A{n}*TT; gn = gn';
%     gd(1:In(n)*R(n)) = gn(:);
% end
% 
% Hnn = AtA{N};
% for n = N-1:-1:1
%     Hnn = kron(Hnn,AtA{n});
% end
% H(In(mode)*R(mode)+1:end,In(mode)*R(mode)+1:end) = Hnn;
%  
% % Gradient grad(f)/grad(vec(G)
% gG = A{mode}.'*Xp.data - AtA{mode}.'*GtAAn;
% gG = reshape(gG,[R(mode) prod(R(1:mode-1)) prod(R(mode+1:N))]);
% gG = permute(gG,[2 1 3]);
% gd(In(mode)*R(mode)+1:end) = gG(:);
% g = gd;
% 
% %%
% v = [reshape(A{mode}',[],1); G(:)];
% 
% ivn= v;
% alphan = nan(N+1,1);
% 
% for n = mode
%     vn = v(1:In(n)*R(n));
%     gdn = gd(1:In(n)*R(n));
%     msk = (vn~=0) & (gdn~= 0);
%     alphan(n) = max([eps,min(-gdn(msk).*vn(msk))]);
%     ivn(1:In(n)*R(n)) = alphan(n)./vn;
% end
% vn = v(end-pR+1:end);
% gdn = gd(end-pR+1:end);
% msk = (vn~=0) & (gdn~= 0);
% alphan(N+1) = max([eps,min(-gdn(msk).*vn(msk))]);
% ivn(end-pR+1:end) = alphan(N+1)./vn;
% 
% gdreg2 =  gd + ivn;
% Hreg = H;
% dH = H(1:size(H,1)+1:end) ;
% Hreg(1:size(H,1)+1:end) = dH(:) + ivn./v + mu;
% 
% d = Hreg\gdreg2;
% 
% % [G2,A2] = update(G,A,d,R,In,N);
% In = In(:)';R = R(:)';
% for n = mode
%     dd = d(1:In(n)*R(n)');
%     A{n} = A{n} + reshape(dd,R(n),[])';
%     A{n} = max(eps,A{n});
% end
% G(:) = G(:) + d(In(mode) * R(mode)+1:end);
% G(:) = max(eps,G(:));
% 
% alpha([mode N+1]) = alphan([mode end]);
% end

