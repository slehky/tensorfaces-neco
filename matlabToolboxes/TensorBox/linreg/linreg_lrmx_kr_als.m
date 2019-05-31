function [X,fcost] = linreg_lrmx_als(y,U,R,opts)
%
% Alternating algorithm for the linear regression 
%
%       min \| y - khatrirao(U1,U2)' * vec(X) \|^2 + mu/2 * \|X\|_F^2
%
% where X = X1*X2' is of rank R and size I1 x I2, 
%
%  U is a cell array of 2 feature vector U{n}
%
%  U{n} = [u_n1, ..., u_nK] : is of size I_n x K
%
%
% Parameters
%
%    init: 'nvec'  initialization method
%    maxiters: 200     maximal number of iterations
%    mu: damping parameter
%
%    printitn: 0
%    tol: 1.0000e-06
%
% Phan Anh Huy, 2017
%


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    X = param; return
end


%%
if param.printitn ~=0
    fprintf('\nAlternating Single Core Update Algorithm for Single-term Multivariate Polynomial Regression\n');
    fprintf(' min  sum_k | y_k -  < X , (uk1 o uk2 o ... o ukN)> |^2\n')
    fprintf(' subject to : X = A * B''')
end

K = numel(y);
N = numel(U);
I = cellfun(@(x) size(x,1),U)';

damping = param.damping;

% Initialization
if isa(param.init,'ktensor')
    X = param.init;
    X = X.U;
elseif iscell(param.init)
    X = param.init;
    % elseif isa(param.init,'TTeMPS')
    %     X = param.init;
else
    switch param.init
        case 'rand'
            X = arrayfun(@(n) randn(n,R),I,'uni',0)';
            X = cellfun(@(x) x/norm(x,'fro'),X,'uni',0);
    end
end

X{1} = orth(X{1});


%% Iterate the main algorithm

Z = reshape(cell2mat(cellfun(@(x,y) x'*y,X(:),U,'uni',0)'),R,[],N);

fcost = 0;
cnt = 0;
% cleanupfunc = onCleanup(@cleanUp);
eps_ = 1e-7;


% largescale = K > max(R*I)*5;
largescale= false;

%Gamma = cellfun(@(x) x'*x,X,'uni',0);
for kiter = 1:param.maxiters
    % update U2
    cnt = cnt +1;
    
    for n = [2 1]
        % alternating update X1 and X2
        if n == 2
            m = 1;
        else
            m = 2;
        end
        
        %    min \| y - P^T * vec(X) \|^2 + mu/2 * \|X\|_F^2  (1)
        % P = khatrirao(U2,U1)
        % w: is the weigth vector, can be 1s
        % In order to update X1, X2 is orthogonalized, then  
        % solve the sub-problem for X1 
        %    min \| y -  Pn^T * vec(X1) \|^2 + mu * \|X1\|_F^2
        %
        %% least squares
        if largescale == false
            Pn = khatrirao(Z(:,:,m),U{n});
            
            % full computation to update xn
            Pny = Pn*y;
            % solve the sub-problem (2b) min | y - Pn*x|^2 + mu/2 |x|^2
            
            % closed-form update
            xn = solve_x(Pn,Pny,damping);
            Xn = reshape(xn,[],R); 
           
        else
            % This might be useful for very large scale 
             %RUN the parallel proximal algorithm 
            xn = ppa_solver(U{n},Z(:,:,m),X{n},K,I(n),R,damping);
            Xn = reshape(xn,[],R);
            
            if I(n) > R % Un In x K, Zm: R x K
                % Pn'*x
                Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n}'*reshape(x,I(n),[]),K,R,[]),Z(:,:,m)'),2),K,[]);%
            else
                %Ptx = @(x) reshape(sum(U{n}.*(reshape(x,I(n),[])*Z(:,:,m)),1)',[],1) ;
                Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n},1,I(n),K),reshape(reshape(x',[],R)*Z(:,:,m),[],I(n),K)),2),[],K)';
            end
            
            
            %             % for large scale problem, solve the nonlinear optimization to
            %             % avoid computing Pn*Pn'
            %
            %             if I(n) > R % Un In x K, Zm: R x K
            %                 % Pn*x
            %                 Px = @(x) reshape(U{n} * khatrirao(x',Z(:,:,m))',[],size(x,2));
            %                 % Pn'*x
            %                 Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n}'*reshape(x,I(n),[]),K,R,[]),Z(:,:,m)'),2),K,[]);%
            %             else
            %                 Px = @(x) reshape(khatrirao(U{n},x') * Z(:,:,m)',size(x,2),[])';
            %                 %Ptx = @(x) reshape(sum(U{n}.*(reshape(x,I(n),[])*Z(:,:,m)),1)',[],1) ;
            %                 Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n},1,I(n),K),reshape(reshape(x',[],R)*Z(:,:,m),[],I(n),K)),2),[],K)';
            %             end
            %             options = optimoptions('fminunc','Algorithm','trust-region',...
            %                 'Display','off','GradObj','on',...
            %                 'Hessian','user-supplied','HessMult',@(H,v) HessMultFcn(H,v,Px,Ptx,damping));
            %             Pny = Px(y);
            %             [xn,fval] = fminunc(@(x) myfun(x,Px,Ptx,Pny,damping),X{n}(:),options);
            %
            %             Xn = reshape(xn,[],R);
            
            %% quasi-newton
            %
            %             if I(n) > R % Un In x K, Zm: R x K
            %                 % Pn*x
            %                 Px = @(x) reshape(U{n} * khatrirao(x',Z(:,:,m))',[],size(x,2));
            %                 % Pn'*x
            %                 Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n}'*reshape(x,I(n),[]),K,R,[]),Z(:,:,m)'),2),K,[]);%
            %             else
            %                 Px = @(x) reshape(khatrirao(U{n},x') * Z(:,:,m)',size(x,2),[])';
            %                 %Ptx = @(x) reshape(sum(U{n}.*(reshape(x,I(n),[])*Z(:,:,m)),1)',[],1) ;
            %                 Ptx = @(x) reshape(sum(bsxfun(@times,reshape(U{n},1,I(n),K),reshape(reshape(x',[],R)*Z(:,:,m),[],I(n),K)),2),[],K)';
            %             end
            %             Pny = Px(y);
            %
            %             options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off','GradObj','on');
            %             fungrad = @(x) myfun(x,Px,Ptx,Pny,damping);
            %             [xn2,fval] = fminunc(fungrad,X{n}(:),options);


        end
        
         
        % cost function value
        if n == 1
            if largescale == false
                fcost(cnt) = norm(y - Pn'*Xn(:))^2/2;
            else
                fcost(cnt) = norm(y - Ptx(Xn(:)))^2/2;
            end
            if damping ~=0
                %fcost(cnt) = fcost(cnt) + damping * trace(Qn * (Xn'*Xn));
                fcost(cnt) = fcost(cnt) + damping/2 * trace((Xn'*Xn)); % since Qn is an identity matrix
            end
        end
        
        % orthogonalise
        [QQ,RR] = qr(Xn,0);
        % check and truncate the rank
        ss = sum(abs(RR).^2,2);
        ixs = ss > eps_*sum(ss);
        R = sum(ixs); %  % rank may change
        X{n} = QQ(:,ixs);X{m} = X{m}*RR(ixs,:)';
        % if
        Z(ixs,:,n) = X{n}'*U{n};
        Z(ixs,:,m) = RR(ixs,:)*Z(:,:,m);
        Z = Z(ixs,:,:);
        
        % update Gamma
        %Gamma = cellfun(@(x) x'*x,X,'uni',0);
    end
    
    %
    if mod(kiter,param.printitn)==0
        fprintf('(%d,%d)- %d\n',kiter,cnt,fcost(cnt))
    end
    
    
    % check stopping criteria
    if (cnt > N) && (abs(fcost(cnt) - fcost(cnt-N))<param.tol)
        break
    end
end


%%
   
end



%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addParameter('init','rand',@(x) (iscell(x) || isa(x,'ktensor')||...
    isa(x,'TTeMPS') || ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'exac'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
% param.addOptional('compression',true);
% param.addOptional('compression_accuracy',1e-6);
% param.addOptional('noise_level',1e-6);% variance of noise
param.addOptional('printitn',0);
param.addOptional('normX',[]);
param.addOptional('damping',0);
param.addParameter('compression',true);   % 0 1 


param.parse(opts);
param = param.Results;


end
% 

%%
function xn = solve_x(Pn,Pny,damping)
% 
%  x'*(Pn*Pn' + mu I)*x - 2*y'*x  
PnPn = Pn*Pn';
if damping ~= 0
    PnPn(1:size(PnPn,1)+1:end) = PnPn(1:size(PnPn,1)+1:end) + damping;
end 
xn = PnPn\Pny;
end

%%
function xn = ppa_solver(Un,Zm,xn,K,In,R,damping)
% Parallel proximal to solve 
%    min \| y - Pn^T * x \|^2 + mu/2 \|x\|^2
% 
%  where Pn is long matrix 
%
idx = 1:K;
% split the K entries into blocks of IJ
blksize = min(K/2,In*R*20);
no_blks = floor(K/(blksize));
ix_blks = [1:blksize:K-blksize K+1];

% Parallel Proximal algorithm
F = cell(no_blks,1);
mu = damping/no_blks;
for kblk = 1:no_blks
    i_kblk = idx(ix_blks(kblk):ix_blks(kblk+1)-1);
    Pn_k = khatrirao(Zm(:,i_kblk),Un(:,i_kblk));
    param_k.y = y(i_kblk);
    param_k.A = @(x) Pn_k'*x;
    param_k.At = @(x) Pn_k*x;
    Pnky = Pn_k*y(i_kblk);
    PnPnk = Pn_k*Pn_k';
    F{kblk}.eval = @(x) 1/2*norm(param_k.y - param_k.A(x))^2 + mu/2*norm(x)^2;
    %F{kblk}.prox = @(x,gamma)  prox_l2(x, gamma, param_k);
    F{kblk}.prox = @(x,gamma)  (gamma*(PnPnk)+(gamma*mu+1)*eye(size(Pn_k,1)))\(x + gamma * Pnky);
end
ppa_param.gamma = 1;
ppa_param.do_ts = @(x) log_decreasing_ts(x, 10, 0.1, 80);
[xn, info] = ppxa(xn(:), F,ppa_param);
end

function [f,g] = myfun(x,Px,Ptx,y,mu)
% y = Px(y)
% f = 1/2*x'*Pn*Pn'*x + mu * x'*x - y'*x
% Ptx = Pn'*x;
% Px = Pn*x
Pnx = Ptx(x);
f = 1/2*(Pnx'*Pnx)+ mu/2*(x'*x) - y'*x;
%g = Pn*Pnx + mu*x - y;
g = Px(Pnx) + mu*x - y;
end

% % Since Qn is always an identity matrix
% % the expression is simplified
% function [f,g,H] = myfun(x,Px,Ptx,y,mu)
% % f = 1/2*x'*Pn*Pn'*x + mu * x'*x - y'*x
% % Ptx = Pn'*x;
% % Px = Pn*x
% Pnx = Ptx(x);
% f = 1/2*(Pnx'*Pnx)+ mu/2*(x'*x) - y'*x;
% %g = Pn*Pnx + mu*x - y;
% g = Px(Pnx) + mu*x - y;
% %H = Pn*Pn'+mu*eye(size(Pn,1));%
% H = 1;
% end
% 
% function w = HessMultFcn(H,v,Px,Ptx,mu)
% % H = Pn*Pn'+mu*eye(In*R)
% % if size(v,2) == 1
% w = Px(Ptx(v))+ mu*v;
% % else
% %     w = [Px(Ptx(v(:,1))) Px(Ptx(v(:,2)))]+mu*v;
% % end
% end