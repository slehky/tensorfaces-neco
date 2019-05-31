function [sol, infos,objectiv] = bpdn_conj(Y,options)
% BPDN with conjugate condition.
%
% Sparse Approximation with conjugate condition
%
% This algorithm approximates a matrix Y by a (nonnegative) matrix X of
% rank-R by solving the optimization problem
%
%    min || X ||_1    
%    subject to || Y - X ||_F^2 < epsilon
%    and F'*vec(X) = 0
%
% where || X ||_1 denotes the ell-1 norm of the matrix X.
%
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
    F = 0;  
else
    [F,foe] = qr(F,0);
end

param.maxit = param.maxiters;
   
Y = options.A(Y);% forward transform
if F~=0
    F = options.A(reshape(F,SzY));% forward transform
    F = F(:);
    % vec(Yf_tf)'* vec(y) = 0
    Abp = @(x) x - F*(F'*x);
else
    Abp = @(x) x;
end



%% BPDN with noise conjugate
% addpath('/Users/phananhhuy/Documents/MATLAB/TFOCS/TFOCS-1.3.1')
% normY = norm(Y(:));
sigma_noise = options.epsilon;
%noise_level = min(normY*.95,max(normY*.90,sqrt(nnz(Y(:)))*sigma_noise)); % noise power which should be gradually reduced.
noise_level = sqrt(nnz(Y(:)))*sigma_noise; % noise power which should be gradually reduced.


[sol,infos,objectiv] = solve_bpdn_conj(Y(:), noise_level, Abp, param);
sol = Abp(sol);
sol =  options.At(reshape(sol,size(Y))); % inverse transform

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
param.addOptional('abs_tol',1);
param.addOptional('epsilon',inf); % noise sigma

param.parse(opts);
param = param.Results;
end

 
 