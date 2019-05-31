function [sol, infos,objective] = lasso_cj(Y,options)
% This is Lasso with some conjugate condition.
%
% Sparse Approximation with conjugate condition
%
% This algorithm approximates a matrix Y by a nonnegative matrix X of
% rank-R by solving the optimization problem
%
%    min 1/2 || Y - X ||_F^2 + gamma * || X ||_1    (1)
%    subject to F'*vec(X) = 0
%
% where || X ||_* denotes the nuclear norm of the matrix X.
%
%
% The second problem is solved using the proximal splitting method
%
%     min f1(X) + f2(x)  
%
% with
%     f1(X) = 1/2 || Y - X ||_F^2  s.t F'*vec(X) = 0   
%           prox_{lda*f1}(Z) = (I - F*F') * (lda*Y + Z)/(1+2*lda)
%     f2(X) = gamma * ||X||_1
%           prox_{lda * f2}(Z)= Soft(Z,lda)
%
% See also an ADMM algorithm sparse_c.m
%
% Phan Anh-Huy, September, 2015.
%
if ~exist('options','var'), options = struct; end
param = parseInput(options);
if nargin == 0
    sol = param; return
end
 
F =[];
if isfield(param,'F') && ~isempty(param.F)
    F = param.F;
end
if isfield(param,'Yref') && ~isempty(param.Yref)
    F = param.Yref;
end
if isempty(F);
    F = 0; 
else
    [F,foe] = qr(F,0); % subspace of F
end
%Q = eye(size(F,1)) - F*F';
Q = @(x,idx) lasso_linoper(F,x,idx);
y= Y(:);

tau=param.tau;

% 
% % solving the problem
% if ~isempty(param.init)
%     x = param.init;
% else
%     x = zeros(size(y));
% end

% update x by solving lasso
%  min  lambda * |x|_1 + 1/2 \|y - F*F'*x|_2^2

[x,infos] = mlasso(Q,y,'NumLambda',5,'sizeX',[numel(y) numel(y)]);
[objective,id] = min(infos.MSE);
sol = x(:,id);

% [x,infos] = mlasso(Q,y,'CV',4,'sizeX',[numel(y) numel(y)]);
% sol = x(:,infos.IndexMinMSE);
% sol = reshape(sol,size(Y));
% objective = infos.MSE(infos.IndexMinMSE); 

end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init',[],@(x) (isnumeric(x) || '' ||...
    ismember(x(1:4),{'zero' 'data'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('tau',1);
param.addOptional('gamma',1);
param.addOptional('A',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('At',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('F',[]); % linear operator in F'*vec(X) = 0
param.addOptional('Yref',[]); % linear operator in F'*vec(X) = 0
param.addOptional('kron_unfolding_size',[]); % linear operator in F'*vec(X) = 0
param.addOptional('size',[]);
param.addOptional('autothresh',true);
param.addOptional('abs_tol',0);

param.parse(opts);
param = param.Results;
end


function Q = lasso_linoper(F,x,idx)
if (nargin < 3) || isempty(idx)
    Q = x - F* (F'*x);
elseif ~any(idx<0)
    Q = -F * (F(idx,:)'*x(idx));
    Q(idx) = x(idx) + Q(idx);
else % idx<0
    idx = abs(idx);
    Q = x(idx) - F(idx,:)* (F'*x);
end
end
 