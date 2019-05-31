function [X,out,cost,cone_d] = linreg_lrmx_ref(y,A,szX,delta,mask,z,opts)
% Solving the least squares minimization
%
%   min rank(X) + mu * \| X \|_F^2  + 1/2*|z - A*vec(X)|^2
%
%
%  subject to   \| y - A_omega*vec(X)\| < delta
%
%  where y is a vector of length K,
%  A can be a structure matrix A = khatrirao(V,U)'.
%          or an numerical matrix of size IJ x K
% A_omega = A(mask,:);
%
%
% Phan Anh Huy, 24/03/2017
%

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    X = param; return
end

K = numel(y);

%% COmpression of K exceeds number of elements of X
%  A = Q*F
%  \| y - A*x\| = \|Q'*y - F*x\|
compression = (K > 2*prod(szX)) && (param.compression == true);


if compression
    % Find subspace of A
    % compression when K >> I*J
    if isstruct(A)
        Af = khatrirao(A.V,A.U)';
        Afomega = Af(mask(:),:);
    end
    
    
    y0 = y;
    z0 = z;
    delta0 = delta;
    
    
    % compress A'*x = V*Ax*x
    Q = Af'*Af;
    [Vq,Sq] = eig(Q);
    Sq = max(diag(Sq),0);
    Sq = sqrt(Sq);
    
    % |z - A*x| = |V'*z - Ax*x|
    if min(Sq) == 0
        iss = find(Sq>1e-8);
        z = (Vq(:,iss)'*(Af'*z))./Sq(iss); % compressed y
        Az = diag(Sq(iss))*Vq(:,iss)';
    else
        z = (Vq'*(Af'*z))./(Sq);
        Az = diag(Sq)*Vq';
    end
    % define linear operator
    
    Azmap = @(X) Az*X(:);% A* X(:)
    AzTmap = @(b) reshape(Az'*b,szX); % A'*b
    
    % compress A'*x = V*Ax*x
    Q = Afomega'*Afomega;
    [Vq,Sq] = eig(Q);
    Sq = max(diag(Sq),0);
    Sq = sqrt(Sq);
    
    
    % |y - Aomeaga*x| = |V'*y - Ax*x|
    if min(Sq) == 0
        iss = find(Sq>1e-8);
        y = (Vq(:,iss)'*(Afomega'*y))./Sq(iss); % compressed y
        Ay = diag(Sq(iss))*Vq(:,iss)';
    else
        y = (Vq'*(Afomega'*y))./(Sq);
        Ay = diag(Sq)*Vq';
    end
    % define linear operator
    
    Aymap = @(X) Ay*X(:);% A* X(:)
    %AyTmap = @(b) reshape(Ay'*b,szX); % A'*b
    
    % adjust the delta
    %   \|y - Aomega*x\|^2 = \|yx - Ax*x\|^2 + \|y\|^2 - \|yx|^2
    delta = sqrt(delta0^2 - norm(y0)^2 + norm(y)^2);
    
else
    if isstruct(A)
        % Amap   = A*X(:)
        Azmap = @(X) sum(A.U.*(X*A.V),1)';
        AzTmap = @(b) reshape(A.U*bsxfun(@times,A.V',b),szX);% a vector
        
        Aymap = @(X) sum(A.U(:,mask(:)).*(X*A.V(:,mask(:))),1)';
        
    elseif isnumeric(A)
        Azmap = @(X) A*X(:);% A* X(:)
        AzTmap = @(b) reshape(A'*b,szX); % A'*b
        Aymap = @(X) A(mask,:)*X(:);% A* X(:)
    end
end

X0  = param.init;

if iscell(X0)
    X0 = X0{1}*X0{2}';
end

if ischar(param.init)
    switch param.init
        case 'empty'
            X0  = [];
            
        case 'lowrank1'  % low rank to reshape(A'*b.szX)
            G=ATmap(b);
            err = norm(G - Amap(G));
            if err < delta
                X0 = G;
            else
                [u,s,v] = svd(G);s = diag(s);
                Rr = min(szX);
                AF = zeros(K,Rr);
                for r = 1:Rr
                    AF(:,r) = Amap(u(:,r) * v(:,r)');
                end
                Qf = AF'*AF;
                b2 = AF'*b;
                err_r = zeros(Rr,1);
                for r = 1:Rr
                    err_r(r) = s(1:r)'*Qf(1:r,1:r)*s(1:r) - 2*b2(1:r)'*s(1:r);
                end
                R = find(err_r<delta^2,1,'first');
                X0 = u(:,1:R) * diag(s(1:R))*v(:,1:R)';
            end
    end
end

out = [];
switch param.method
    case {'sdpt' 'cvx'} %       via CVX
        %         global cvx___;
        mu = param.mu;
        
        cvx_begin
            cvx_precision best
            variable X(szX(1),szX(2))
            if param.mu == 0
                minimize norm_nuc(X) + 1/2*sum_square(z- Azmap(X))
            else
                minimize norm_nuc(X) + mu/2 * sum_square(vec(X)) + 1/2*sum_square(z- Azmap(X))
            end
            subject to
            norm(Aymap(X) - y ) <= delta
        cvx_end
        out = cvx_optval;
        
        %         cvx___.extra_params = [];
        
        %     case 'tfocs'
        %         %  minimize norm_nuc(X) + 0.5*mu*norm(X-X0,'fro').^2
        %         %        s.t.     ||A * x - b || <= epsilon
        %
        %
        %         if isempty(param.mu) | (param.mu==0)
        %             param.mu = 1;
        %         end
        %
        %         % for TFOCCS
        %         %ATmap2 = @(y) reshape(ATmap(y),szX);% same size of X
        %         tf_opts = [];
        %         tf_opts.alg    = 'AT'; % gradien descend 'AT'  'LLM'; 'GRA' N83
        %         tf_opts.maxIts = param.maxiters;
        %         tf_opts.continuation = (param.continuation == 1) || (param.continuation == true);
        %
        %         z0 =[];
        %         %z0 = Amap(X0)-y;
        %         Aop = linop_handles({ szX [numel(y), 1]}, Amap, ATmap, 'R2R');
        %         [X, out, opts ] = solver_sNuclearDN(Aop, y, delta, param.mu, [], z0, tf_opts );
        %
        % %         optval =  norm_nuclear(X) + param.mu/2*norm(X,'fro')^2;
        % %         cond_K = norm(Amap(X) - y );
        %
        % %         vec = @(x) x(:);
        % %         mat = @(x) reshape(x,szX);
        % %
        % %         if compression
        % %             z0 = Amap(X0)-yx;
        % %             %Aop = linop_handles({ szX [prod(szX), 1]}, Amap, ATmap, 'R2R');
        % %
        % % %             Aop   = linop_matrix( Ax, 'R2R' ); % n1 x n2 is size of x
        % % %             It  = linop_handles({ szX,[prod(szX),1] }, vec, mat, 'R2R' );
        % % %             Aop   = linop_compose( Aop, It );
        % %
        % %             [X, out, opts ] = solver_sNuclearDN2(Aop, yx, deltax, param.mu, [], z0, tf_opts );
        % %         else
        % %             z0 = Amap(X0)-y;
        % %             %Aop = linop_handles({ szX [K, 1]}, Amap, ATmap, 'R2R');
        % %
        % %             Aop   = linop_matrix( A, 'R2R' ); % n1 x n2 is size of x
        % %             It  = linop_handles({ szX,[K,1] }, vec, mat, 'R2R' );
        % %             Aop   = linop_compose( Aop, It );
        % %
        % %
        % %             [X, out, opts ] = solver_sNuclearDN2(Aop, y, delta, param.mu, [], z0, tf_opts );
        % %         end
        
end

cost = cvx_optval;
if compression
    cost = cost + (norm(z0)^2 - norm(z)^2)/2;
end
cone_d = norm(y-Aymap(X));
if compression
    cone_d = sqrt(cone_d^2 + norm(y0)^2 - norm(y)^2);
end
end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.StructExpand = true;
param.addParameter('init','empty',@(x) isnumeric(x) || iscell(x) || ismember(x,{'rand' 'ppa' 'empty'}));
param.addParameter('maxiters',2000);
param.addParameter('tol',1e-6);
param.addParameter('verbose',0);
param.addParameter('mu',0); %in mu * |X - X0|_F^2
param.addParameter('gamma',1); %parameter used in ADMM algrithm
param.addParameter('method','sdpt',@(x) ismember(x(1:3),{'sdp' 'cvx' 'ppa' 'tfo' 'adm'}));    % sdpt ppa  tfocs
param.addParameter('continuation',1);   % 0 1
param.addParameter('compression',true);   % 0 1

param.parse(opts);
param = param.Results;
end
