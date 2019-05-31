function [sol, infos,objectiv] = sparsify_orth_transform(Y,options)
% Lasso with conjugate condition.
%
% Sparse Approximation with conjugate condition
%
% This algorithm approximates a matrix Y by a (nonnegative) matrix X of
% rank-R by solving the optimization problem
%
%    min  || X ||_1  
% 
%   subject to   ||Y - X||_F^2 < epsilon
%
%  epsilon often indicates the noise variance.
%
% Phan Anh-Huy, September, 2015.
%
if ~exist('options','var'), options = struct; end
param = parseInput(options);

if nargin == 0
    sol = param; return
end

SzY = size(Y); 
 
F =[];
if isfield(param,'F') && ~isempty(param.F)
    F = param.F;
end
if isfield(param,'Yref') && ~isempty(param.Yref)
    F = param.Yref;
end
if isempty(F);
    F = 0; % it should be an identity matrix, but set to 1.
else
    [F,foe] = qr(F,0);
end
 

%% Define the prox of f2 see the function proj_B2 for more help
patchsize = param.kron_unfolding_size;
if ~isempty(patchsize)
    Yh = kron_unfoldingN(Y,patchsize);
else
    Yh = Y;
end

if param.reduceDC
    vecOfMeans = mean(Yh,2);
    Yh = bsxfun(@minus,Yh,vecOfMeans);
end
sigma = param.epsilon;
errT = sigma*1.1;
v = param.A(Yh)';
% Coefs = sparse_vec(v,errT); % sparse coefficients
Coefs = sparse_vec2(v,errT); % sparse coefficients

Yh = param.At(Coefs');
if param.reduceDC
    Yh = bsxfun(@plus,Yh, vecOfMeans);
end

if ~isempty(patchsize)
    Yh = kron_foldingN(Yh,patchsize);
end

f1_val = norm(Coefs(:),1);
f2_val = norm(Y(:) - Yh(:))^2;

sol = Yh;
infos = struct('iter',1,'f1_val',f1_val,'f2_val',f2_val);
objectiv = [];

end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init',[],@(x) (isnumeric(x) || '' ||...
    ismember(x(1:4),{'zero' 'data'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);% will be used as standard deviation  of noise: sigma
param.addOptional('printitn',0);
param.addOptional('tau',1); 
param.addOptional('gamma',1);
param.addOptional('A',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('At',@(x) x); % linear operator in || vec(Y) - A vec(X)
param.addOptional('F',[]); % linear operator in F'*vec(X) = 0
param.addOptional('Yref',[]); % linear operator in F'*vec(X) = 0
param.addOptional('kron_unfolding_size',[]); % patch size
param.addOptional('size',[]);
param.addOptional('autothresh',true);
param.addOptional('abs_tol',1);
param.addOptional('reduceDC',1);
param.addOptional('epsilon',inf);

param.parse(opts);
param = param.Results;
end

 


