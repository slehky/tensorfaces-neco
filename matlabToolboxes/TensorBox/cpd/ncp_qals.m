function [P,output] = ncp_qals(X,R,opts)
% Fast Recursive ALS algorithm for nonnegative CP factorizes the N-way
% nonnegative tensor X  into nonnegative factors of R components. 
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
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
%     .TraceMSAE: check mean square angular error as stoping condition
%
% OUTPUT: 
%  P:  ktensor of estimated factors
%  output:  
%      .Fit
%      .NoIters 
%
% EXAMPLE
%   X = tensor(rand([10 20 30]));  
%   opts = ncp_qals;
%   opts.init = {'nvec' 'random' 'random' 'fiber'}; % multi-initialization
%   [P,output] = ncp_qals(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "On Fast Computation of Gradients
% for CP Algorithms", 2011 
% [2] Matlab Tensor toolbox by Brett Bader and Tamara Kolda
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
% 
% See also: cp_als, cp_fastals, ncp_hals
%
% TENSOR BOX, v1. 2012
% Copyright Phan Anh Huy, 2011, 2012
% 2012, Fast CP gradient


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
onCleanup(@setoutput);
N = ndims(X); normX = norm(X);I = size(X);
%% Initialize factors U
Uinit = ncp_init(X,R,param); U = Uinit;
fprintf('\nFast nonnegative CP_QALS:\n');

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
end

% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(I); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
ns = find(Jn<=Kn,1,'last');
updateorder = [ns:-1:1 ns+1:N];

%%
if param.verify_convergence == 1
    %lambda = ones(R,1);
    %P = ktensor(U);
    %err=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
    %fit = 1-sqrt(err)/normX; 
    fit = 0;
    %if param.TraceFit
    %    output.Fit = fit;
    %end
    if param.TraceMSAE
        msae = (pi/2)^2;
    end
end

UtU = zeros(R,R,N);
for n = 1:N
    UtU(:,:,n) = U{n}'*U{n};
end
lambda = sqrt(diag(prod(UtU,3)))';

%% Main Loop: Iterate until convergence
Pmat = [];Niniter = 5;
for iter = 1:param.maxiters
    
    if param.verify_convergence==1
        Uold = U;
        if param.TraceFit, fitold = fit;end
        if param.TraceMSAE, msaeold = msae;end
    end
    
    % Iterate over all N modes of the tensor
    for n = updateorder(1:end)
        
        %         % METHOD 1
        %         U{n} = bsxfun(@times,U{n},lambda);
        %         T2 = mttkrp(Y,U,n);
        %         T32 = prod(UtU(:,:,setdiff(1:N,n)),3);
        %
        %         Util = T2/T32 ;
        %         actidx = Util<0;
        %
        %         for kss = 1:R-1
        %             %                     fprintf('%d\t%d\t%d\t%d\n',n,ri,jidx,sum(actidx(:)))
        %             Util(actidx) = 0;
        %             actrows = ~all(~actidx,2);
        %             for i = find(actrows)'
        %                 idxi = ~actidx(i,:);
        %                 %                         idxi =  ~idxi;
        %                 uj = T2(i,idxi) - U{n}(i,:) * T32(:,idxi) +...
        %                     U{n}(i,idxi) * T32(idxi,idxi);
        %                 W = T32(idxi,idxi);
        %                 %dbstop if error
        % %                 if det(W) > 1e-10
        %                     uj = uj/W;
        % %                 else
        %                     %continue
        % %                     uj = bsxfun(@rdivide,uj,diag(W)');
        % %                 end
        %                 Util(i,idxi) = max(0,uj);
        %                 %Util(i,idxi) = min(Util(i,idxi),uj);
        %             end
        %             actidx = Util<0;
        %         end
        %
        %         U{n} = max(eps,Util);
        %         lambda = sqrt(sum(U{n}.^2));
        %         U{n} = bsxfun(@rdivide,U{n},lambda+eps);
        %         UtU(:,:,n) = U{n}'*U{n};
        
        
        % METHOD 2
        U{n} = bsxfun(@times,U{n},lambda);
        
        if (N==2) || isa(X,'ktensor')
            Phi = mttkrp(X,U,n);
        elseif isa(X,'tensor') || isa(X,'sptensor')
            if (n == ns) || (n == ns+1)
                [Phi,Pmat] = cp_gradient(U,n,X);
            else
                [Phi,Pmat] = cp_gradient(U,n,Pmat);
            end
            %[Phi,Pmat] = cp_gradient(U,n,Pmat);
        end
        G = Phi;
        
        Q = prod(UtU(:,:,setdiff(1:N,n)),3);
        
        if param.rowopt == 0 % Update all component
            Phi = Phi';
            v = nqp(kron(eye(I(n)),Q),Phi(:));
            U{n} = reshape(v,R,I(n))';
        else % Update row by row
            Util = Phi/Q;
            actidx = Util<0;
            Util(actidx) = 0;
            actrows = ~all(~actidx,2);
            for ir = find(actrows)'
                uj = nqp(Q,Phi(ir,:)');
                Util(ir,:) = max(0,uj);
            end
            U{n} = max(eps,Util);
        end
        %        %met2
        %         if rcond(Q) > 1e-10
        %             Util = Phi/Q;
        %             Phi = Phi';
        %             IdxP = Util<0;
        %         else
        %             Phi = Phi';
        %             IdxP = Phi<0;
        %         end
        %         Qq = spalloc(I(n)*R,I(n)*R,R^2*I(n));
        %         for ki = 1:I(n)
        %             Qq(R*(ki-1)+1:R*ki,R*(ki-1)+1:R*ki) = Q;
        %         end
        %         v = nqp(Qq(~IdxP(:),~IdxP(:)),Phi(~IdxP(:)));
        %         Un = zeros(R,I(n));
        %         Un(~IdxP(:)) = v;
        %         U{n} = Un';
        
        % Innerproduct for fast computation of approximation error
        if param.TraceFit && (n == updateorder(end))
            innXXhat = sum(sum(U{updateorder(end)}.*G));
        end
        
        lambda = sqrt(sum(U{n}.^2));
        U{n} = bsxfun(@rdivide,U{n},lambda+eps);
        UtU(:,:,n) = U{n}'*U{n};
    end
    
    if param.verify_convergence==1
        if param.TraceFit 
            normresidual = sqrt( normX^2 + lambda*prod(UtU,3)*lambda' - 2*innXXhat);
            fit = 1 - (normresidual / normX); %fraction explained by model
            fitchange = abs(fitold - fit);
            stop(1) = fitchange < param.tol;
            if nargout >=2
                output.Fit = [output.Fit; iter fit];
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
P = ktensor(lambda',U);

% Normalize factors and fix the signs
P = arrange(P);%P = fixsigns(P);

if (param.printitn>0) && (param.verify_convergence==1)
    %normresidual = sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,(P)) );
    %fit = 1 - (normresidual / normX); %fraction explained by model
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
                    Pmat = reshape(Pmat.data,[],prod(I(right))); % Right-side projection
                elseif isa(Pmat,'sptensor')
                    Pmat = reshape(Pmat,[prod(size(Pmat))/prod(I(right)),prod(I(right))]); % Right-side projection
                    Pmat = spmatrix(Pmat);
                else
                    Pmat = reshape(Pmat,[],prod(I(right))); % Right-side projection
                end
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
%                 if (isempty(KRP_2) && (numel(left) > 2))
%                     [KRP_left,KRP_2] = khatrirao(A(left));
%                 elseif isempty(KRP_2)
%                     KRP_left = khatrirao(A(left));
%                     %KRP_2 = [];
%                 else
%                     KRP_left = KRP_2; KRP_2 = [];
%                 end
                T = reshape(Pmat,prod(I(left)),I(n),[]);
                if R>1
                    T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
                    T = sum(T,1);
                    %G = squeeze(T);
                    G = reshape(T,[],R);
                else
                    G = (KRP_left'*T)';
                end
            else
                %G = squeeze(Pmat);
                G = reshape(Pmat,[],R);
            end
            
        elseif n >=ns+1
            if n ==ns+1
                if numel(left) == 1
                    KRP_left = A{left}';
                elseif numel(left) > 1
                    KRP_left = khatrirao_t(A(left));
                    %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
                else 
                    KRP_left = 1;
                end
                if isa(Pmat,'tensor')
                    T = reshape(Pmat.data,prod(I(left)),[]);
                elseif isa(Pmat,'sptensor')
                    T = reshape(Pmat,[prod(I(left)) prod(size(Pmat))/prod(I(left))]); % Right-side projection
                    T = spmatrix(T);
                else
                    T = reshape(Pmat,prod(I(left)),[]);
                end
                %
                Pmat = KRP_left * T;   % Left-side projection
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
                
                if (n == (ns+1)) && (numel(right)>=2)
                    %KRP_right = KRP_right0;
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right0',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T, R,[])';
                    else
                        %G = squeeze(T) * KRP_right0;
                        G = reshape(T,[],prod(I(right))) * KRP_right0;
                    end
                else
                    KRP_right = khatrirao(A(right));
                    if R>1
                        T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
                        T = sum(T,3);
                        %G = squeeze(T)';        % Right-side projection
                        G = reshape(T,R,[])';        % Right-side projection
                    else
                        %G = squeeze(T) * KRP_right;
                        G = reshape(T,I(n),[]) * KRP_right;
                    end
                end
            else
                %G = squeeze(Pmat)';
                G = reshape(Pmat,R,[])';
            end
            
        end
        
%         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
%             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
%             size(KRP_left,1),size(KRP_left,2),...
%             size(KRP_right,1),size(KRP_right,2))
    end


    function setoutput
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
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||ismember(x(1:4),{'rand' 'nvec' 'fibe'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addParamValue('rowopt',true,@islogical);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);
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
        Uinit = param.init;
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

%%
function x = nqp(Q,b)
% Algorithm solves the NQP problem: f(x) = 1/2 x^T Q x - b^T x
% Phan Anh Huy , 08/2010
%Idx0 = find(b>0);Idx = Idx0;
Idx0 = 1:numel(b);Idx = Idx0;
x = zeros(size(b))-1;
while ~isempty(Idx0)
    x(Idx) = Q(Idx,Idx)\b(Idx);
    Idx0 = find(x(Idx) < 0);
    %dbstop if warning
    Idx(Idx0) = [];
    %             if numel(Idx) >0
    %                 1
    %             end
end
x = max(eps,x);
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
