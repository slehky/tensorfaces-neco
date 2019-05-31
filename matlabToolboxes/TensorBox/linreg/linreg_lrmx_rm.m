function [X,xcost,info] = linreg_lrmx_rm(y,A,R,szX,opts)
% Algorithm for the linear regresssion problem 
%
%           min  |y - A'*vec(X)|^2 + mu/2 * |X|_F^2
%           subject to  rank(X) = R
%
%  y is an input vector, 
%  A can be a matrix or a Khatri-Rap structured matrix with two factors U
%  and V  such that A = khatrirao(V,U)'.
% 
%  X is on the Riemaninan manifold of low-rank matrix 
%
%  This problem is solved using the trust-region or conjugate-gradient
%  algorithm on the Riemanian manifold.
%
%
% Phan Anh Huy, 24/03/2017.
%
% addpath(genpath('/Users/phananhhuy/Documents/Matlab/Manifold/manopt'))


if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    X = param; return
end
param.verbosity = param.verbose;


damping = param.damping;

% Create the problem structure.
%clear manifold problem;
manifold = fixedrankembeddedfactory(szX(1),szX(2),R);
problem.M = manifold;


if ischar(param.init)
    switch param.init
        case {'empty' 'rand'}
            param.init  = [];
    end
elseif iscell(param.init)
    if numel(param.init) == 3
        param.init = struct('U',param.init{1},'S',param.init{2},'V',param.init{3});
    elseif numel(param.init) == 2
        [U,ru] = qr(param.init{1},0);
        [V,rv] = qr(param.init{2},0);
        [u,s,v] = svd(ru*rv');
        param.init = struct('U',U*u,'S',s,'V',V*v);
    end
end

% convert cell arrary to structure 
if iscell(A)
   A = struct('U',A{1} ,'V',A{2});
end
%%
K = numel(y);
compression = (K > 2*prod(szX)) && (param.compression == true);
if compression   
    % Find subspace of A
    % compression when K >> I*J
    if isstruct(A)
        A = khatrirao(A.V,A.U);
    end
    Q = A*A';
    [Vq,Sq] = eig(Q);
    Sq = sqrt(diag(Sq));
    %Uq = A*Vq'/Sq;
    normy2 = norm(y)^2;
    y = (Vq'*(A*y))./Sq;% new y 
    A = Vq*diag(Sq);  % and new A
end

% %% ALS
% alsopts = linreg_lrmx_als;
% alsopts.maxiters = 1;
% [X,xcost2] = linreg_lrmx_als(y,A,R,szX,alsopts);

%%

%Ay = A'*y;
if isstruct(A)
    if size(A.U,1) < size(A.V,1)
        Atb = @(b) reshape(bsxfun(@times,A.U,b')*A.V',[],1);
    else
        Atb = @(b) reshape(A.U*bsxfun(@times,A.V',b),[],1);
    end
    %A*X
    Ax = @(X) ((A.V'*X.V).*(A.U'*X.U))*diag(X.S);
    
elseif isnumeric(A)
    
    Atb = @(b) A*b; % A'*b
    Ax = @(X)  A'*vec(X.U*X.S*X.V'); % A*vec(X.U*X.S*X.V')
    
end

    
%A'*A*x
AtAx = @(x) Atb(Ax(x));

% precompute A'*y
Ay = Atb(y);

% Define the problem cost function and its gradient.
% problem.cost  = @(X) 1/2*norm(y - Ax(X))^2;
% problem.egrad = @(X) reshape(-Ay+AtAx(X),szX);
if damping ==0
    problem.cost  = @(X) fcostgrad(X,'cost');
    problem.egrad = @(X) fcostgrad(X,'grad');
else
    problem.cost  = @(X) fcostgrad(X,'cost',damping);
    problem.egrad = @(X) fcostgrad(X,'grad',damping);
end
problem.ehess = @(X,H) reshape(AtAx(problem.M.tangent2ambient(X, H)),szX);

% Execute the solver on Riemanian manifold
% [X, xcost, info] = trustregions(problem,param.init,param); 
[X, xcost, info] = conjugategradient(problem,param.init,param); 

if compression
    % adjust the cost values
    xcost = xcost + (normy2 - norm(y)^2)/2;
end

function [out] = fcostgrad(X,cmode,damping)
    persistent Xold costc gradc;
    if nargin < 3
        damping = 0;
    end
    
    if ischar(X) || isempty(Xold) || (size(X.U,1) ~= size(Xold.U,1)) ||  (size(X.V,1) ~= size(Xold.V,1))
        Xold = X;

        yh = Ax(X);
        costc = 1/2*norm(y - yh)^2;
        gradc = reshape(-Ay+Atb(yh),szX); 
     
        if damping ~= 0
            costc = costc + damping/2 * norm(diag(X.S))^2;
            gradc = gradc + damping * X.U*X.S*X.V';
        end
    else
        % X == Xold
        errX =  norm(diag(Xold.S))^2 + norm(diag(X.S))^2 - 2* diag((X.U'*Xold.U) *  (Xold.S) * (Xold.V'*X.V))'*diag(X.S);
        if abs(errX) >=1e-12 
            Xold = X;
            yh = Ax(X);
            costc = 1/2*norm(y - yh)^2;
            gradc = reshape(-Ay+Atb(yh),szX);
            
            if damping ~= 0
                costc = costc + damping/2 * norm(diag(X.S))^2;
                gradc = gradc + damping * X.U*X.S*X.V';
            end
            
        end
         
    end
    
    switch cmode
        case 'cost'
            out = costc;
        case 'grad'
            out = gradc;
    end
    
end


end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.StructExpand = true;
param.addParameter('init','empty',@(x) iscell(x) || isstruct(x) || (ischar(x) && ismember(x,{'rand' 'ppa' 'empty'})));
% param.addOptional('init','empty',@(x) (iscell(x) || (ischar(x) && ismember(x,{'rand' 'ppa' 'empty'}))));
param.addParameter('maxiter',2000);
param.addParameter('tol',1e-8);
param.addParameter('verbose',0); 
param.addParameter('damping',0); 
param.addParameter('compression',true); 

% param.addOptional('mu',0);   %in mu * |X - X0|_F^2
% param.addOptional('method','sdpt',@(x) ismember(x(1:3),{'sdp' 'ppa' 'tfo'}));    % sdpt ppa  tfocs
% param.addOptional('continuation',1);   % 0 1 
param.parse(opts);
param = param.Results;
end