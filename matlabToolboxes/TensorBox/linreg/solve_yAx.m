function X = solve_yAx(y,A,szX,delta, opts)
% Solving the least squares minimization 
%           | y  - A*(X)| <= delta
%  subject to the matrix X is with minimal rank 
%  y is a vector 
%  A can be a structure matrix A = khatrirao(V,U)'.
%          or an numerical matrix of size IJ x K  
%
% 
% Problem 1 : An equivalent optimization problem 
% 
%           minimize     rank(X)    
%           subject to   | y  - A(X)| <= delta
%
% or 
%
%           minimize     nuclear_norm(X)   
%           subject to   | y  - A(X)| <= delta
%
% which can be solved using the PPA algorithm or SPDT through CVX.
% 
% Problem 2:  A variant problem with an addition constraint on X
%
%           minimize     nuclear_norm(X) + mu * \| X \|_F^2 
%           subject to   | y  - A(X)| <= delta
%
%  can be solved using the SDPT through CVX or TFOCS.
%
%
% Problem 3:  When X is with fixed rank 
%   
%           minimize    | y - A(X)|^2  
%           subject to  rank(X) = R
% 
%  This problem is solved using the trust-region algorithm on the Riemanian
%  manifold. See solve_yAx_fixedrank.
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
        A = khatrirao(A.V,A.U)';
    end
    Q = A'*A;
    [Vq,Sq] = eig(Q);
    Sq = sqrt(diag(Sq));
    %Uq = A*Vq'/Sq;
    yx = (Vq'*(A'*y))./Sq;
    Ax = diag(Sq)*Vq'; 
 
    % define linear operator 
    
    Amap = @(X) Ax*X(:);% A* X(:)
    ATmap = @(b) reshape(Ax'*b,szX); % A'*b
    % adjust the delta 
    %   \|y - A*x\|^2 = \|yx - Ax*x\|^2 + \|y\|^2 - \|yx|^2 
    deltax = sqrt(delta^2 - norm(y)^2 + norm(yx)^2);

else
    if isstruct(A)
        % Amap   = A*X(:)
        Amap = @(X) sum(A.U.*(X*A.V),1)';
        % Atmap = A'*y
        %ATmap = @(b) reshape(A.U*bsxfun(@times,A.V',b),[],1);% a vector
        ATmap = @(b) reshape(A.U*bsxfun(@times,A.V',b),szX);% a vector
    elseif isnumeric(A)
        Amap = @(X) A*X(:);% A* X(:)
        ATmap = @(b) reshape(A'*b,szX); % A'*b
    end
end


switch param.init 
    case 'empty'
        param.init  = [];
end
    
if param.mu == 0
    param.method = 'sdpt';
end


switch param.method
    case 'ppa'
        %  min          sum(svd(X))
        %  subject to   norm(b1-A1(X)) <= delta        
        param.plotyes = 0;
       
        if compression 
            [X,iter,time,sd,hist] = dualPPA(Amap,ATmap,yx,delta,szX(1),szX(2),szX(1)*szX(2),0,0,param);
        else
            % [Xppa,iter,time,sd,hist] = dualPPAold(Amap,ATmap,b,delta,I,J,m1,m2,m3,param);
            [X,iter,time,sd,hist] = dualPPA(Amap,ATmap,y,delta,szX(1),szX(2),K,0,0,param);
        end
        

    case {'sdpt' 'cvx'} %       via CVX
        if param.mu == 0
            
            if compression
                cvx_begin
                    cvx_precision best
                    variable X(szX(1),szX(2))
                    minimize norm_nuc(X)
                    subject to
                    norm(Amap(X) - yx ) <= deltax
                cvx_end
            else
                
                cvx_begin
                    cvx_precision best
                    variable X(szX(1),szX(2))
                    minimize norm_nuc(X)
                    subject to
                    norm(Amap(X) - y ) <= delta
                cvx_end
            end
            
        else
            mu = param.mu;
            % x0 = zeros(szX);
            if compression
                
                cvx_begin
                cvx_precision best
                    variable X(szX(1),szX(2))
                    minimize norm_nuc(X) + mu/2 * sum_square(vec(X))  
                    subject to
                    norm(Amap(X) - yx ) <= deltax
                cvx_end
                
            else
                cvx_begin
                cvx_precision best
                    variable X(szX(1),szX(2))
                    minimize norm_nuc(X) + mu/2 * sum_square(vec(X))  
                    subject to
                    norm(Amap(X) - y ) <= delta
                cvx_end
            end
        end

        
    case 'tfocs'
        %  minimize norm_nuc(X) + 0.5*mu*norm(X-X0,'fro').^2
        %        s.t.     ||A * x - b || <= epsilon


        if isempty(param.mu) | (param.mu==0)
            param.mu = 1;
        end
        
        % for TFOCCS
        %ATmap2 = @(y) reshape(ATmap(y),szX);% same size of X
        tf_opts = [];
        tf_opts.alg    = 'GRA'; % gradien descend 'AT'  'LLM';
        tf_opts.maxIts = param.maxiters;
        tf_opts.continuation = (param.continuation == 1) || (param.continuation == true);
        
        if compression
            Aop = linop_handles({ szX [prod(szX), 1]}, Amap, ATmap, 'R2R');
            [X, out, opts ] = solver_sNuclearDN(Aop, yx, deltax, param.mu, param.init, [], tf_opts );
        else
            Aop = linop_handles({ szX [K, 1]}, Amap, ATmap, 'R2R');
            [X, out, opts ] = solver_sNuclearDN(Aop, y, delta, param.mu, param.init, [], tf_opts );
        end
 
                
end
  

end
 

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.StructExpand = true;
param.addParameter('init','empty',@(x) ismember(x,{'rand' 'ppa' 'empty'}));
param.addParameter('maxiters',2000);
param.addParameter('tol',1e-6);
param.addParameter('verbose',0);  
param.addParameter('mu',0); %in mu * |X - X0|_F^2
param.addParameter('method','sdpt',@(x) ismember(x(1:3),{'sdp' 'cvx' 'ppa' 'tfo'}));    % sdpt ppa  tfocs
param.addParameter('continuation',1);   % 0 1 
param.addParameter('compression',true);   % 0 1 

param.parse(opts);
param = param.Results;
end
 