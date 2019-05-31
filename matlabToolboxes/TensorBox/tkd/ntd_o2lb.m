function [T,output] = ntd_o2lb(Y,R,opts)
% An alternating algorithm updates factor matrices and core tensor in a
% Nonnegative Tucker Decomposition.
%
% The update rules are based on the Gauss-Newton algorithm with log-barrier
% function to impose nonnegativity constraints.
% The algorithm is a simplified version of the NTD_LM algorithm to update
% only one factor matrix at each step. 
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
%          opts.init = {'random' 'random' 'nvec'};
%          See details in ntd_init.m.
%
%     .printitn: Print fit every n iterations {1}
%
% OUTPUT: 
%  T:  ttensor of estimated factors and core tensor
%  output:  
%      .Fit
%      .NoIters 
%
% EXAMPLE
%   X = tensor(randn([10 20 30]));  
%   opts = ntd_o2lb;
%   opts.init = {'nvec' 'nmfs' 'random'};
%   [P,output] = ntd_o2lb(X,5,opts);
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
% See also: ntd_init, ntd_hals3, ntd_lm
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
initopts = ntd_init;
initopts.init = param.init;
initopts.alsinit = param.alsinit;
initopts.maxiters = 10;

if N == 3    
    initopts.ntd_func = @ntd_hals3;
end
 
% [A,G] = ntd_init(Y,R,initopts);
% G = tensor(G);
Yhat = initopts.ntd_func(Y,R,initopts);
A = Yhat.u; G = Yhat.core;
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
fprintf('\n GN algorithm with log-barrier function for NTD:\n');

fit = 0;
fitarr = [];
alpha = mean(G(:).^2);

warning off;
%% Main Loop: Iterate until convergence
flagtol = 0;

AtA = cellfun(@(x) x'*x , A,'uni',0);
options = optimset('Display','none','TolFun',1e-8);
for iter = 1:param.maxiters
    pause(0.001)
    fitold = fit;

    for n = 1:N
        %% Update An
        % Hessian = kron(eye(In), Hn)
        Gn = double(tenmat(G,n));
        GtAA = ttm(G,AtA,-n);
        GtAAn = double(tenmat(GtAA,n));
        Hn = GtAAn * Gn';
        
        %% gradient Gn = grad(f)/grad(vec(A{n}')
        Xp = ttm(Y,A,-n,'t');Xp = tenmat(Xp,n);
        gAn = Xp.data * Gn.' - A{n}*Hn; %gAn = gAn';
        
        % select parameter of the log-barier function
        msk = (A{n}~=0) & (gAn~= 0);
        alphan = max([eps,min(-gAn(msk).*A{n}(msk))]);
        alpha(n) = alphan;
        ivn = alphan./(A{n}+eps);
         
        gAn_reg =  gAn + ivn;
        
        ivn = ivn./(A{n}+eps);
        for ik = 1:In(n)
            dn = (Hn + diag(ivn(ik,:)))\gAn_reg(ik,:)';
            A{n}(ik,:) = A{n}(ik,:) + dn';
            
%             z = gAn_reg(ik,:)' + (Hn + diag(ivn(ik,:)))*A{n}(ik,:)';
%             x2 = lsqlin((Hn + diag(ivn(ik,:))),z,-eye((R(n))),zeros(R(n),1),[],[],[],[],A{n}(ik,:)',options);
%             A{n}(ik,:) = x2';
        end
        A{n} = max(eps,A{n});
        AtA{n} = A{n}' * A{n}; 
        
        %% Update core
        HG = AtA{N};
        for n2 = N-1:-1:1
            HG = kron(HG,AtA{n2});
        end
        
        % Gradient grad(f)/grad(vec(G)
        XtA = A{n}.'*Xp.data;
        GAtA = AtA{n}.'*GtAAn;
        gG = XtA - GAtA;
        gG = reshape(gG,[R(n) prod(R(1:n-1)) prod(R(n+1:N))]);
        gG = permute(gG,[2 1 3]);
        gG = gG(:);
        
        % log-barier parameter for the core tensor
        msk = (G(:)~=0) & (gG~= 0);
        alphan = max([eps,min(-gG(msk).*G(msk))]);
        alpha(N+1) = alphan;
        ivn = alphan./(G(:)+eps);
        
        gGreg =  gG + ivn;
        ivn = ivn./(G(:)+eps);
        dH = HG(1:size(HG,1)+1:end);
        HG(1:size(HG,1)+1:end) = dH(:) + ivn;
        
%         dG = HG\gGreg;
%         G(:) = G(:) + dG;
%         G = tensor(max(eps,G.data));

        z = gGreg + HG*G(:);
        x2 = lsqlin(HG,z,-eye(size(HG)),zeros(size(z)),[],[],[],[],G(:),options);
        G(:) = x2;


%         Jinfo = speye((prod(R)));
%         options = optimset('JacobMult',...
%             @(Jinfo,Y,flag)kron_inv(Jinfo,Y,flag,AtA,ivn),'TolFun',0);
% 
%         z = gGreg + HG*G(:);
%         x2 = lsqlin(Jinfo,z,-eye(size(HG)),zeros(size(z)),[],[],[],[],[],options);
%         G(:) = x2;
        
    end
    
    % Cost value
%     [err2,errest] = costvalue(Y,G,A,alpha,normY);
    XtA = XtA';
    GAtA = HG*G(:);
    errest = normY^2 + (GAtA(:) - 2*XtA(:)).' * G(:);
    
%     cost = errest;
%     for n = 1:N
%         cost = cost - alpha(n) * sum(log(A{n}(:)));
%     end
%     cost = cost - alpha(N+1) * sum(log(G(:)));
   

    for n=1:N
        am=sqrt(sum(A{n}.^2));          % normalization
        A{n}=bsxfun(@rdivide,A{n},am);
        G = ttm(G,diag(am),n);
        AtA{n} = A{n}' * A{n};
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

% % %%
% function [err,err0] = costvalue(X,G,A,alpha,normX)
% N = ndims(X);
% XA = ttm(X,A,'t');
% GAtA = full(ttm(ttensor(G,A),A,'t'));
% err0 = normX^2 + (GAtA(:) - 2*XA(:)).' * G(:);
% %Xhat = ttensor(G,A);
% %err0=normX.^2 + norm(Xhat).^2 - 2 * innerprod(X,Xhat);
% %err0 = double(err0);
% % ldlog = cellfun(@(x) sum(log(x(:))), A);
% 
% %err = err0-alpha*(sum(ldlog) + sum(log(G(:))));
% if numel(alpha) == 1
%     alpha = alpha(ones(N+1),1);
% end
% err = err0;
% for n = 1:N
%     err = err - alpha(n) * sum(log(A{n}(:)));
% end
% err = err - alpha(n+1) * sum(log(G(:)));
% end


function w = kron_inv(Jinfo,Y,flag,A,d)
% This function computes the Jacobian multiply functions
%  C = kron(An) + diag(d)
% C*Y, C'*Y, C'*C*Y
R = cellfun(@(x) size(x,2),A);
Y = full(Y);
C = kron(A{3},kron(A{2},A{1})) + diag(d);
if (flag > 0)
    w = C*Y;%w = Jpositive(Y);
elseif (flag < 0) % C * Y and % C'*Y
    w = C*Y;%w = Jpositive(Y);
else % C'*C*Y
    w = C*C*Y;
    %w = Jpositive(Jpositive(Y));
end

    function w = Jpositive(Y)
        Yt = reshape(full(Y),[R,size(Y,2)]);
        w = ttm(tensor(Yt),A,1:numel(A));
        w = reshape(double(w),[],size(Y,2));
        w = w + bsxfun(@times,Y,d);
    end

end