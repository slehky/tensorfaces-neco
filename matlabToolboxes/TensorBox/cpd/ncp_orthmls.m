function [P,output] = cp_orthmls(X,R,opts)
% Fast Multiplicative ALgorithm for nonnegative CP factorizes the N-way
% nonnegative tensor X  into nonnegative factors of R components. 
%
% INPUT:
%   X:  N-D data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'|cell array]
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%     .TraceFit: check fit values as stoping condition.
%     .TraceMSAE: check mean square angular error as stoping condition
% Output: 
%  P:  ktensor of estimated factors
%
% REF:
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "On Fast Computation of Gradients
% for CP Algorithms", 2011 
% [2] Matlab Tensor toolbox by Brett Bader and Tamara Kolda
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
% 
% See also: cp_als
%
% Copyright 2011, Phan Anh Huy.


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
N = ndims(X); normX = norm(X);I = size(X);
param.lortho(end+1:N) = 0;
affmode = param.affmode;
%% Initialize factors U
fprintf('\nFast nonnegative CP_multiplicative LS:\n');
Uinit = ncp_init(X,R,param); U = Uinit;

%% Output
if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% Permute tensor dimension (tranpose) so that I1<=I2<= ... <= IN
p_perm = [];
if ~issorted(I)
    [I,p_perm] = sort(I);
    X = permute(X,p_perm);
    U = U(p_perm);
    param.lortho = param.lortho(p_perm);
    
    if affmode ~= 0 
        affmode = find(p_perm == affmode);
    end
end

% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(I); Kn = Jn(end)./Jn;
ns = find(Jn>Kn,1);
if ((ns >= (N-1)) && (ns > 2))
    ns = ns -1;
end
if param.lortho(end) == 1           % change order of updates to fix bug occuring for
    updateorder = [ns+1:N ns:-1:1];     % orthogonal nCP
else
    updateorder = [ns:-1:1 ns+1:N];
end

%%
if param.verify_convergence == 1
    lambda = ones(N,1);
    P = ktensor(U);
    err=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
    fit = 1-sqrt(err)/normX; 
    if param.TraceFit
        output.Fit = fit;
    end
    msae = (pi/2)^2;
end

UtU = zeros(R,R,N);
for n = 1:N
    UtU(:,:,n) = U{n}'*U{n};
end


%% Affine or offset part
if affmode ~= 0 
    sumX = sum(double(tenmat(X,affmode)),2);
    pI = prod(I([1:affmode-1 affmode+1:N]));
    
    aU = cell(N,1);
    aU{affmode} = sumX/pI;
    for n = 1:N
        aU{n} = ones(I(n),1);
    end
    aP = ktensor(aU);
end

%% Main Loop: Iterate until convergence
Pmat = [];Niniter = 5;
lambda = ones(1,R);
for iter = 1:param.maxiters
    
    if param.verify_convergence==1
        fitold = fit;Uold = U;
        msaeold = msae;
    end
    %% Affine part
    if affmode ~= 0
        sumU = ones(1,R);
        for n = [1:affmode-1 affmode+1:N]
            sumU = sumU.* sum(U{n},1);
        end
        aP.U{affmode} = aP.U{affmode} .* sumX ./ ...
            (aP.U{affmode} * pI + U{affmode} * sumU');
    end
    %% Iterate over all N modes of the tensor
    for n = updateorder(1:end)
        %U{n} = bsxfun(@times,U{n},lambda);
        %UtU(:,:,n) = U{n}'*U{n};
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        if isa(X,'ktensor')
            G = mttkrp(X,U,n); 
        elseif isa(X,'tensor')
           [G,Pmat] = cp_gradient(U,n,Pmat);
        end

        % Compute the matrix of coefficients for linear system
        
        for inn = 1:Niniter
            if affmode ~= 0
                aG = mttkrp(aP,U,n);
            else
                aG = 0;
            end
            if param.lortho(n) ~= 0
%                 Gamma = prod(UtU(:,:,setdiff(1:N,n)),3);
%                 den = G + U{n} * (Gamma * UtU(:,:,n)) ;
%                 num = U{n} * Gamma + U{n} * (G' * U{n}) + eps;
%                 U{n} = U{n} .* den ./num;
                

                U{n} = max(eps,U{n} .* (G + U{n} * (aG' * U{n})) ./...
                    (U{n} * (G' * U{n}) + aG + eps));
                %U{n} = max(eps,U{n} .* (G ./(U{n} * (U{n}' * G)  + eps)));
            else
                U{n} = max(eps,U{n} .* G ./...
                ( U{n} * prod(UtU(:,:,setdiff(1:N,n)),3) + aG + eps ));
            end
            
            UtU(:,:,n) = U{n}'*U{n};
        end
        if n~= ns
            lambda = sum(U{n});
            U{n} = bsxfun(@rdivide,U{n},lambda);
            U{ns} = bsxfun(@times,U{ns},lambda);
            UtU(:,:,ns) = U{ns}'*U{ns};
            UtU(:,:,n) = U{n}'*U{n};
        end
        
    end
    
    lambda = ones(R,1);
%     for n = [1:ns-1: ns+1:N]
%         lambda = lambda.*sum(U{n})';
%         U{n} = bsxfun(@rdivide,U{n},sum(U{n}));
%         UtU(:,:,n) = U{n}'*U{n};
%     end
%     U{ns} = bsxfun(@times,U{ns},lambda');
%     UtU(:,:,ns) = U{ns}'*U{ns};
%     lambda = ones(R,1);

    
    if param.verify_convergence==1
        if param.TraceFit 
            P = ktensor(lambda(:),U);
            if affmode ~= 0
                P = P + aP;
            end
            normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,  (P)) );
            fit = 1 - (normresidual / normX); %fraction explained by model
            fitchange = abs(fitold - fit);
            stop(1) = fitchange < param.tol;
            if nargout >=2
                output.Fit = [output.Fit; fit];
            end
        end
        
        if param.TraceMSAE 
            msae = SAE(U,Uold);
            msaechange = abs(msaeold - msae); % SAE changes
            stop(2) = msaechange < param.tol;
            if nargout >=2
                output.MSAE = [output.MSAE; msae];
            end
            
        end
        
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: ',iter);
            if param.TraceFit
                fprintf('fit = %e fitdelta = %7.1e ', fit, fitchange);
            end
            if param.TraceMSAE
                fprintf('msae = %e delta = %7.1e', msae, msaechange);
            end
            fprintf('\n');
        end
        
        % Check for convergence
        if (iter > 1) && any(stop)
            break;
        end
    end
end

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = ktensor(lambda(:),U);
if affmode ~= 0
    P = P + aP;
end

% Normalize factors and fix the signs
P = arrange(P);%P = fixsigns(P);

if param.printitn>0
    normresidual = sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,(P)) );
    fit = 1 - (normresidual / normX); %fraction explained by model
    fprintf(' Final fit = %e \n', fit);
end

% Rearrange dimension of the estimation tensor 
if ~isempty(p_perm)
    P = ipermute(P,p_perm);
    %[foe,ip_perm] = sort(p_perm);
    % Uinit = Uinit(ip_perm);
end
if nargout >=2
    output.NoIters = iter;
end

%% CP Gradient with respect to mode n
    function [G,Pmat] = cp_gradient(A,n,Pmat)
        right = N:-1:n+1; left = n-1:-1:1;
        %KRP_right =[];KRP_left = [];
        if n <= ns
            if n == ns
                if numel(right) == 1
                    KRP_right = A{right};
                elseif numel(right) > 1
                    KRP_right = khatrirao(A(right));
                else
                    KRP_right = 1;
                end
                
                Pmat = reshape(X.data,[],prod(I(right))); % Right-side projection
                Pmat = Pmat * KRP_right ;
            else
                Pmat = reshape(Pmat,[],I(right(end)),R);
                if R>1
                    Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],I(right(end)),R));
                    Pmat = sum(Pmat,2);    % fast Right-side projection
                else
                    Pmat = Pmat * A{right(end)};
                end
            end
            
            if ~isempty(left)       % Left-side projection
                KRP_left = khatrirao(A(left));
                T = reshape(Pmat,prod(I(left)),I(n),[]);
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
                    T = sum(T,1);
                    G = squeeze(T);
                else
                    G = (KRP_left'*T)';
                end
            else
                G = squeeze(Pmat);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left};
                elseif numel(left) > 1
                    KRP_left = khatrirao(A(left));
                else 
                    KRP_left = 1;
                end
                T = reshape(X.data,prod(I(left)),[]);
                Pmat = KRP_left' * T;   % Left-side projection
            else
                if R>1
                    Pmat = reshape(Pmat,R,I(left(1)),[]);
                    Pmat = bsxfun(@times,Pmat,A{left(1)}');
                    Pmat = sum(Pmat,2);      % Fast Left-side projection
                else
                    Pmat = reshape(Pmat,I(left(1)),[]);
                    Pmat = A{left(1)}'* Pmat;
                end
            end
            
            if ~isempty(right)
                T = reshape(Pmat,[],I(n),prod(I(right)));
                KRP_right = khatrirao(A(right));
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
                    T = sum(T,3);
                    G = squeeze(T)';        % Right-side projection
                else
                    G = squeeze(T) * KRP_right;
                end
            else
                G = squeeze(Pmat)';
            end
            
        end
        
%         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
%             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
%             size(KRP_left,1),size(KRP_left,2),...
%             size(KRP_right,1),size(KRP_right,2))
    end
end


% %% Khatri-Rao xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function krp = khatrirao(A,B)
% if nargin==2
%     R = size(A,2);
%     krp = zeros(size(A,1)*size(B,1),R);
%     for r = 1:R
%         d = B(:,r) * A(:,r)';
%         krp(:,r) = d(:);
%     end
% else
%     
%     krp = A{1};
%     I = cellfun(@(x) size(x,1),A);
%     R = size(A{1},2);
%     for k = 2:numel(A)
%         temp = zeros(size(krp,1)*I(k),R);
%         for r = 1:R
%             d = A{k}(:,r) * krp(:,r)';
%             temp(:,r) = d(:);
%         end
%         krp = temp;
%     end
% end
% end
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function K = khatrirao(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1};
for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
end
K = reshape(K,[],R);
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

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addParamValue('lortho',0);
param.addParamValue('lsparse',0);
param.addParamValue('lsmooth',0);
param.addParamValue('affmode',0);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',false,@islogical);
param.addParamValue('TraceMSAE',true,@islogical);
param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end

%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function Uinit = ncp_init(X,R,param)
% Set up and error checking on initial guess for U.
N = ndims(X);
if iscell(param.init)
    Uinit = param.init;
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
        Uinit{n} = abs(real(nvecs(X,n,R)));
    end
else
    error('Invalid initialization');
end
end

function [msae,msae2,sae,sae2] = SAE(U,Uh)
% Square Angular Error
% sae: square angular error between U and Uh  
% msae: mean over all components
% 
% [1] P. Tichavsky and Z. Koldovsky, Stability of CANDECOMP-PARAFAC
% tensor decomposition, in ICASSP, 2011, pp. 4164?4167. 
%
% [2] P. Tichavsky and Z. Koldovsky, Weight adjusted tensor method for
% blind separation of underdetermined mixtures of nonstationary sources,
% IEEE Transactions on Signal Processing, 59 (2011), pp. 1037?1047.
%
% [3] Z. Koldovsky, P. Tichavsky, and A.-H. Phan, Stability analysis and fast
% damped Gauss-Newton algorithm for INDSCAL tensor decomposition, in
% Statistical Signal Processing Workshop (SSP), IEEE, 2011, pp. 581?584. 
%
% Phan Anh Huy, 2011

N = numel(U);
R = size(U{1},2);
sae = zeros(N,size(Uh{1},2));
sae2 = zeros(N,R);
for n = 1: N
    C = U{n}'*Uh{n};
    C = C./(sqrt(sum(abs(U{n}).^2))'*sqrt(sum(abs(Uh{n}).^2)));
    C = acos(min(1,abs(C)));
    sae(n,:) = min(C,[],1).^2;
    sae2(n,:) = min(C,[],2).^2;
end
msae = mean(sae(:));
msae2 = mean(sae2(:));
end