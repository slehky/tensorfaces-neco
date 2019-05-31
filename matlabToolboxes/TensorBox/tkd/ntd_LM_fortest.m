function [T,A,G,fit,iter] = ntd_hals(Y,R,opts)
% HALS NTD algorithm
% INPUT
% Y     :   tensor with size of I1 x I2 x ... x IN
% R     :   size of core tensor R1 x R2 x ... x RN: [R1, R2, ..., RN]
% opts  :   struct of optional parameters for algorithm (see defoptions)
%   .tol:   tolerance of stopping criteria (explained variation)     (1e-6)
%   .maxiters: maximum number of iteration                             (50)
%   .init:  initialization type: 'random', 'eigs', 'nvecs' (HOSVD) (random)
%   .orthoforce:  orthogonal constraint to initialization using ALS
%   .updateYhat:  update Yhat or not, using in Fast Alpha NTD with ell-1
%                 normalization for factors                             (0)
%   .ellnorm:   normalization type                                      (1)
%   .fixsign:  fix sign for components of factors.                      (1)
%
% Copyright of Anh Huy Phan, Andrzej Cichocki
% Ver 1.0 12/2008, Anh Huy Phan
% Ver 1.1 12/2009, Anh Huy Phan

% Set algorithm parameters from input or by using defaults
defoptions = struct('tol',1e-6,'maxiters',50,'init','random',...
    'ellnorm',1,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,...
    'fixsign',0,'Grule','multiplicative');
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

if strcmp(opts.Grule,'local') %multiplicative
    opts.ellnorm = 2;
end

%% Set up and error checking on initial guess for U.
[A,G] = ntd_initialize(Y,opts.init,opts.orthoforce,R);
G = tensor(G);
%%
fprintf('\nLocal NTD:\n');
% Compute approximate of Y
Yhat = ttensor(G,A);
normresidual = sqrt(normY^2 + norm(Yhat)^2 -2*innerprod(Y,Yhat));
fit = 1 - (normresidual/normY);        %fraction explained by model
fprintf(' Iter %2d: fit = %e \n', 0, fit);
% Yr = Y- tensor(Yhat);
%%
fitarr = [];
In = size(Y);
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
mu = 1e4;
alpha = 10;   %%% initial multiplicative constant for the barrier function
%dalpha=exp(100/opts.maxiters*log(0.005));
dalpha = 0.001;

warning off;
printitn = 5;
fitmax = 1-1e-8;fitchangetol = 1e-9;
%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(0.001)
    fitold = fit;
    err = normY.^2 + norm(Yhat).^2 - 2 * innerprod(Y,Yhat);
    err = double(err);
    ldlog = cellfun(@(x) sum(log(x(:))), A);
    %     err = err-alpha*(sum(ldlog));
    err = err-alpha*(sum(ldlog) + sum(log(G(:))));
    
    A1 = A;G1 = G;
    
    %[d Rr ]=gradHess(Y,G,A,mu,alpha);      %%% improved version of the above
    [d Rr ] = fastgradHess2(Y,G,A,mu,alpha);      %%% improved version of the above
    for n = 1:N
        A{n}(:) = A{n}(:) +  d(In(1:n-1)*R(1:n-1)'+1  : In(1:n)*R(1:n)');
        A{n}(:) = max(eps,A{n}(:));
    end
    G(:) = G(:) + d(In * R'+1:end);
    %      AtA = cellfun(@(x) x'*x,A,'uni',0);
    %      G = G.* ttm(Y,A,'t')./ttm(G,AtA);
    G(:) = max(eps,G(:));
    
    Yhat = ttensor(tensor(G),A);
    err0=normY.^2 + norm(Yhat).^2 - 2 * innerprod(Y,Yhat);
    err0 = double(err0);
    err2 = err0-alpha*(sum(ldlog) + sum(log(G(:))));
    %     err2 = err0-alpha*(sum(ldlog) );
    rho=real((err-err2)/(d(:)'*(Rr+mu*d(:))));
    
    
    if err2>err                                        %% The step is not accepted
        mu=mu*nu; nu=2*nu;
        A = A1;G = G1;
    else
        nu=2;
        mu=mu*max([1/3 1-(2*rho-1)^3]);
        for n=1:N
            am=sqrt(sum(A{n}.^2));   %%% new normalization
            A{n}=bsxfun(@rdivide,A{n},am);
            G = ttm(G,diag(am),n);
        end
        
        fit = 1-sqrt(abs(err0))/normY; %fraction explained by model
        fitchange = abs(fitold - fit);
        fitarr = [fitarr fit];
        if mod(iter,printitn)==0
            fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n',...
                iter, fit, fitchange);
        end
        % Check for convergence
        if (fit>= fitmax) ||  ((iter > 1) && (fitchange < fitchangetol)) % Check for convergence
            fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n',...
                iter, fit, fitchange);
            break;
        end
        
    end
    
    
    if rem(iter,5)==0
        alpha=alpha*dalpha;   %%% decrease value of alpha
    end
    
end
%% Compute the final result
T = ttensor(G, A);
fit = fitarr;
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [d gd]= gradHess(X,G,A,mu,alpha)

%
% computes one step in LM iteration, that is S=(F+mu*I)\R
% It does not need to store F, max. dimension required is 3r^2 x 3r^2
% F is Hessian of the criterion augmented by barrier function (sum of
% logarithms of the factor elements) with multiplicative factor alpha
% R is the gradient of the augmented criterion.
%

N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);
cI = In.*R';
cI = cumsum([0 cI]);

% Jacobian
% J = nan();
J = nan(prod(In),In*R);
for n = 1: N
    Pn = permute_vec_new(In,n);
    Jn = ttm(tensor(G),A,-n);
    Jn = double(tenmat(Jn,n))';
    Jn = kron(Jn,eye(In(n)));
    J(:,cI(n)+1:cI(n+1)) = Pn'*Jn;
end

Jn = A{end};
for n = N-1:-1:1
    Jn = kron(Jn, A{n});
end

J = [J Jn];

v = cellfun(@(x) x(:), A,'uni',0);
v = cell2mat(v(:));
v = [v; G(:)];
gd = X - ttm(G,A);

gd = J'*gd(:) + alpha * 1./v;
H = J'*J;
dH = H(1:size(H,1)+1:end) ;
H(1:size(H,1)+1:end) = dH(:) + alpha * (1./v.^2) + mu;

d = H\gd;
end



%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [d gd]= fastgradHess(X,G,A,mu,alpha)

%
% computes one step in LM iteration, that is S=(F+mu*I)\R
% It does not need to store F, max. dimension required is 3r^2 x 3r^2
% F is Hessian of the criterion augmented by barrier function (sum of
% logarithms of the factor elements) with multiplicative factor alpha
% R is the gradient of the augmented criterion.
%
G = tensor(G);
N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);
cI = In.*R';
cI = cumsum([0 cI]);

%% Full Hessian
H2 = cell(N+1,N+1);
AtA = cellfun(@(x) x'*x , A,'uni',0);
for n = 1:N
    %%%  Jn' Jn
    TT = double(ttt(ttm(G,AtA,-n),G,setdiff(1:N,n)));
    H2{n,n} = kron(TT, eye(In(n)));
    
    Prnin = per_vectrans(In(n),R(n));
    AtAx = AtA;
    AtAx(n+1:end+1) = AtAx(n:end);
    AtAx{n} = eye(R(n));
    Gh = tensor(G.data,[R(1:n) ; 1 ; R(n+1:end)]');
    %%%  Jn' Jm
    for m = n+1:N
        %  Fast way without computing Jacobian, avoid Kronkect products
        Pnr = permute_vec_new(R([1:n n:m-1 m+1:end])',n);
        Prmim = per_vectrans(In(m),R(m));
        % Phi_n * B (kronecker products of AtA...)
        W = ttm(Gh,AtAx,-[n n+1 m+1 ]);
        W = ttm(W,A{m},m+1);
        
        G_m = double(tenmat(G,m));
        Hnm2 = zeros(R(n),R(m),In(n),In(m));
        for in = 1: In(n)
            Y = ttm(W,A{n}(in,:)',n+1);
            Ym = tenmat(Y,m+1);
            Ym = double(Ym) * Pnr';
            %             for im = 1: In(m)
            %                 %Yim = tensor(Ym(im,:),[R(1:n) ;R(n:m-1)  ;R(m+1:end)]');
            %                 %Hnm2(:,:,in,im) = ttt(Yim,G,setdiff(1:N,n),setdiff(1:N,m));
            %
            %                 %Yim_n = reshape(Ym(im,:),prod(R(1:n-1)),R(n),[]);
            %                 %Yim_n = permute(Yim_n,[2 1 3]);
            %                 %Yim_n = reshape(Yim_n,R(n),[]);
            %
            %                 Yim_n = reshape(Ym(im,:),R(n),[]);
            %                 Hnm2(:,:,in,im) = Yim_n * G_m';
            %             end
            %             Hnm2(:,:,in,im) = Yim_n * G_m';
            
            Prnim = per_vectrans(In(m),R(n));
            Yin_m = Prnim * reshape(Ym,In(m)*R(n),[]);
            Hinm = Yin_m * G_m';
            Hnm2(:,:,in,:) = permute(reshape(Hinm,R(n),In(m),R(m),1),[1 3 2]);
        end
        
        Hnm2r = permute(Hnm2,[1 3 2 4]);
        
        
        Hnm2r = reshape(Hnm2r,R(n)*In(n),R(m)*In(m));
        Hnm2r = Prnin' * Hnm2r * Prmim;
        
        H2{n,m} = Hnm2r;
        H2{m,n} = Hnm2r';
    end
    %%  Jn' JN+1
    W = ttm(Gh,AtAx,-[n n+1]);
    HnN = nan(In(n) * R(n),prod(R));
    for in = 1: In(n)
        Y = ttm(W,A{n}(in,:)',n+1);
        Y = double(tenmat(Y,n));
        HnN(1+R(n)*(in-1):R(n)*in,:)= Y;
    end
    HnN = Prnin' * HnN;
    H2{n,N+1} = HnN;
    H2{N+1,n} = HnN';
end

Hnn = 1;
for n = N:-1:1
    Hnn = kron(Hnn,AtA{n});
end
H2{N+1,N+1} = Hnn;

H = cell2mat(H2);

%% GRADIENT
Gr = cell(N+1,1);
Xh = ttensor(G,A);
for n = 1:N
    Prnin = per_vectrans(In(n),R(n));
    
    Gn = double(tenmat(G,n));
    
    Xp = ttm(X,A,-n,'t');
    Xp = tenmat(Xp,n);
    
    Xhp= ttm(Xh,A,-n,'t');
    Xhp = tenmat(Xhp,n);
    tt = Gn * (Xp - Xhp)';
    Gr{n} = Prnin' * tt(:);
end

Xp = full(ttm(X,A,'t')) ;
Xhp = full(ttm(Xh,A,'t')) ;
Gr{N+1} = Xp(:) - Xhp(:);

gd = cell2mat(Gr);

%%
v = cellfun(@(x) x(:), A,'uni',0);
v = cell2mat(v(:));
v = [v; G(:)];

gd =  gd + alpha * 1./v;
% H = J'*J;
dH = H(1:size(H,1)+1:end) ;
H(1:size(H,1)+1:end) = dH(:) + alpha * (1./v.^2) + mu;

d = H\gd;
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [d gd]= fastgradHess2(X,G,A,mu,alpha)

%
% computes one step in LM iteration, that is S=(F+mu*I)\R
% It does not need to store F, max. dimension required is 3r^2 x 3r^2
% F is Hessian of the criterion augmented by barrier function (sum of
% logarithms of the factor elements) with multiplicative factor alpha
% R is the gradient of the augmented criterion.
%
G = tensor(G);
N = ndims(X);In = size(X);
R = cellfun(@(x) size(x,2),A);R = R(:);
cI = In.*R';
cI = cumsum([0 cI]);

%% Full Hessian
H2 = cell(N+1,N+1);
AtA = cellfun(@(x) x'*x , A,'uni',0);
for n = 1:N
    %%%  Jn' Jn
    %TT = double(ttt(ttm(G,AtA,-n),G,setdiff(1:N,n)));
    TT = double(tenmat(ttm(G,AtA,-n),n)) * double(tenmat(G,n))';
    H2{n,n} = kron(TT, eye(In(n)));
    
    Prnin = per_vectrans(In(n),R(n));
    AtAx = AtA;
    AtAx(n+1:end+1) = AtAx(n:end);
    AtAx{n} = eye(R(n));
    Gh = tensor(G.data,[R(1:n) ; 1 ; R(n+1:end)]');
    %%%  Jn' Jm
    for m = n+1:N
        %  Fast way without computing Jacobian, avoid Kronkect products
        % Phi_n * B (kronecker products of AtA...)
        W = ttm(Gh,AtAx,-[n n+1 m+1 ]);
        W = ttm(W,A{m},m+1);
        Wnplo = double(tenmat(W,n+1));
        Iw = size(W); 
        Iy = Iw; Iy(n+1) = R(n);
        
        Pnr = permute_vec_new(R([1:n n:m-1 m+1:end])',n);
        Prmim = per_vectrans(In(m),R(m));
        Prnim = per_vectrans(In(m),R(n));
        Pynp1 = permute_vec_new(Iy,n+1);
        Pymp1 = permute_vec_new(Iy,m+1);
        Pp = Pymp1*Pynp1';
        
        Prnrm = per_vectrans(R(m),R(n));
        G_m = double(tenmat(G,m));
%         Hnm2r = zeros(R(n),In(n),R(m),In(m));
%         for in = 1: In(n)
%             %Y = ttm(W,A{n}(in,:)',n+1);
%             %Ym = tenmat(Y,m+1);
%             
%             Y = A{n}(in,:)' * Wnplo;
%             Ym = reshape(Pp *Y(:),In(m),[]);
%             Ym =  Ym * Pnr';
%             Yin_m = Prnim * reshape(Ym,In(m)*R(n),[]);
%             Hinm = Yin_m * G_m';
%             Hnm2r(:,in,:,:) = permute(reshape(Hinm,R(n),In(m),R(m),1),[1 3 2]);
%         end
%         Hnm2r = reshape(Hnm2r,R(n)*In(n),R(m)*In(m));
          %% %  
%         Y = reshape(A{n}',[],1) * Wnplo;
%         Pq = per_vectranst(size(Y,2),R(n));
%         Y = (Pp *Pq') * reshape(Y',[],In(n));
%         
%         Hnm2r = nan(R(n) * In(n), R(m)*In(m));
%         for in = 1: In(n)
%             Ym = reshape(Y(:,in),In(m),[]);
%             Ym =  Ym * Pnr';
%             Yin_m = Prnim * reshape(Ym,In(m)*R(n),[]);
%             Hinm = G_m * Yin_m';
%             Hinm = Prnrm*reshape(Hinm,R(n) * R(m), In(m));
%             Hinm = reshape(Hinm,R(n),[]);
%             Hnm2r((in-1)*R(n)+1:in*R(n),:) = Hinm;
%         end

        %%
        Y = reshape(A{n}',[],1) * Wnplo;
        Pq = per_vectranst(size(Y,2),R(n));
        Y = (Pp *Pq') * reshape(Y',[],In(n));
        
        % Y ?x In to  (Im * In)  x ??
        Y = reshape(Y,In(m),[],In(n));
        Y = permute(Y,[1 3 2]);
        Y = reshape(Y,In(n) * In(m),[]);
        Y = Y * Pnr';
        
        %
        Y = reshape(Y,In(m),In(n),R(n),[]);
        Y = permute(Y,[3 1 2 4]);
        Y = reshape(Y,In(m)*In(n)*R(n),[]);
       
        Hinm = G_m * Y';
        Hinm = Prnrm*reshape(Hinm,R(n) * R(m), []);
        Hnm2r = reshape(Hinm,R(n),[],In(n));
        Hnm2r = permute(Hnm2r,[1 3 2]);
        Hnm2r = reshape(Hnm2r,R(n) * In(n),[]);
        
        %%
        Hnm2r = Prnin' * Hnm2r * Prmim;
        
        H2{n,m} = Hnm2r;
        H2{m,n} = Hnm2r';
    end
    %%  Jn' JN+1
%     W = ttm(Gh,AtAx,-[n n+1]);
%     HnN = nan(In(n) * R(n),prod(R));
%     for in = 1: In(n)
%         Y = ttm(W,A{n}(in,:)',n+1);
%         Y = double(tenmat(Y,n));
%         HnN(1+R(n)*(in-1):R(n)*in,:)= Y;
%     end

%     W = ttm(Gh,AtAx,-[n n+1]);
%     Iy = size(W); Iy(n+1) = R(n);
%     W = double(tenmat(W,n+1));
%     HnN = nan(In(n)*R(n),prod(R));
%     
%     Pynp1 = permute_vec_new(Iy,n+1);
%     Pymp1 = permute_vec_new(Iy,n);
%     Pp = Pymp1*Pynp1';
%     for in = 1:In(n)
%         Y = A{n}(in,:)' * W;
%         Y = reshape(Pp *Y(:),R(n),[]);
%         HnN(1+R(n)*(in-1):R(n)*in,:)= Y;
%     end

    W = ttm(Gh,AtAx,-[n n+1]);
    Iy = size(W); Iy(n+1) = R(n);
    W = double(tenmat(W,n+1));
    
    Pynp1 = permute_vec_new(Iy,n+1);
    Pymp1 = permute_vec_new(Iy,n);
    Pp = Pymp1*Pynp1';
    
    Y = (Prnin*A{n}(:)) * W;
    Pq = per_vectranst(size(Y,2),R(n));
        
    Y = reshape(Y',[],In(n))' * (Pq *Pp');
    HnN = reshape(Y,R(n)*In(n),[]);
    HnN = Prnin * HnN;
    
    HnN = Prnin' * HnN;
    H2{n,N+1} = HnN;
    H2{N+1,n} = HnN';
end

Hnn = 1;
for n = N:-1:1
    Hnn = kron(Hnn,AtA{n});
end
H2{N+1,N+1} = Hnn;

H = cell2mat(H2);

%% GRADIENT
Gr = cell(N+1,1);
Xh = ttensor(G,A);
for n = 1:N
    Prnin = per_vectrans(In(n),R(n));
    
    Gn = double(tenmat(G,n));
    
    Xp = ttm(X,A,-n,'t');
    Xp = tenmat(Xp,n);
    
    Xhp= ttm(Xh,A,-n,'t');
    Xhp = tenmat(Xhp,n);
    tt = Gn * (Xp - Xhp)';
    Gr{n} = Prnin' * tt(:);
end

Xp = full(ttm(X,A,'t')) ;
Xhp = full(ttm(Xh,A,'t')) ;
Gr{N+1} = Xp(:) - Xhp(:);

gd = cell2mat(Gr);

%%
v = cellfun(@(x) x(:), A,'uni',0);
v = cell2mat(v(:));
v = [v; G(:)];

gd =  gd + alpha * 1./v;
% H = J'*J;
dH = H(1:size(H,1)+1:end) ;
H(1:size(H,1)+1:end) = dH(:) + alpha * (1./v.^2) + mu;

d = H\gd;
end