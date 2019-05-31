function [A,X,Yh,rmse] = ktc_nng_sgrp(Y,Ix,R,opts)
% Single group Kronecker tensor decomposition with nonnegativity constraint.
%
% The input tensor Y is approximated as
%
%     Y = A_1 \ox  X_1 + ... + A_R \ox  X_R
%
% where \ox denotes the Kronecker product between two tensors A_r and X_r.
% All tensors X_r have the same size Ix(1) x Ix(2) x ... x Ix(N) (single
% group). 
%
% For multiple group decomposition, see ktc_nng.m.
%
% Input:  
%   Y   :  data tensor with nan respresenting missing entries in Y.
%   Ix  :  row array indicates size of tensors X_r.
%   R   :  number of tensors X_r.
%   opts:  parameters of the decomposition.
%          Run the algorithm without any input to get the default parameter
%          of the algorithm:  
%          opts = ktc_nng_sgrp;
%
% Output
%   A and X are order-(N+1) tensors comprising A_r and X_r, respectively.
%   Yh  :  approximated tensor
%   rmse:  mse 
%
% Ref:
% [1] A.-H. Phan, A. Cichocki, P. Tichavsky, G. Luta, A. Brockmeier,
% Tensor Completion Through Multiple Kronecker Product Decomposition. 2013
% IEEE International Conference on Acoustics, Speech, and Signal Processing
% ICASSP 2013, p. 3233-3237. 
%
% [2] A.-H. Phan, A. Cichocki, P. Tichavsky, D. P. Mandic, and K.Matsuoka,
% On revealing replicating structures in multiway  data: A novel tensor 
% decomposition approach,? in Latent Variable Analysis and Signal
% Separation, vol. 7191 of Lecture Notes in Computer Science, pp. 297-305.
% Springer 2012.
% 
% Copyright Phan Anh Huy 2011-10-1
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by Phan Anh Huy.

if ~exist('opts','var'),  opts = struct; end
param = parseinput(opts);

if nargin == 0
    A = param; return
end


Iy = size(Y); N = ndims(Y);  % size of tensor Y
Ia = bsxfun(@rdivide,Iy,Ix); % size of tensors A




% Rearrange Y to a matrix using the Kronecker tensor unfolding
Y = kron_unfolding(Y,Ix);

% Indicator tensor : 1 for missing data.
Weights = isnan(Y);
Y(Weights) = 0;

% Initialization 
if iscell(param.init)
    A = param.init{1};if iscell(A), A = A{1};end
    X = param.init{2};if iscell(X), X = X{1};end
end

if ischar(param.init) && strcmp(param.init(1:4),'nvec')
    RP = min([size(Y) R]);
    [A,Sg,X] = msvd(Y,RP);
    A =  A*Sg;
    if R > RP
        A(:,end+1:R) = rand(size(Y,1),R-RP);
        A(:,end+1:R) = rand(size(Y,2),R-RP);
    end
    A =  abs(A); X = abs(X);
elseif ischar(param.init) && strcmp(param.init(1:4),'rand')
    A =  rand(size(Y,1),R);
    X =  rand(size(Y,2),R);
end 
nY = norm(Y(:),'fro');


Yh = A*X';Yh(Weights) = 0;
Yh_old = Yh;

%% fprintf('\nSolving by SVT...\n');
rmse = [];rmse_f = [];
for ki  = 1:param.maxiters
    
    % Update A
    % for smoothness
    if param.smoothA  ~= 0
        As = reshape(A,[Ia,R]);
        As = tensor(As);
        for n = 1:numel(Ia)
            if Ia(n)> 2
                d1 = [8 9 6*ones(1,Ia(n)-3) 5]';
                d2 = [-8 -4*ones(1,Ia(n)-2)]';
                d3 = [2 ones(1,Ia(n)-3)]';
                Ln = diag(d1,0) + ...
                    diag(d2,1) + ...
                    diag(d2,-1) +...
                    diag(d3,2) + ...
                    diag(d3,-2);
                %                         Ln = spdiags(d1,0,Iag(n),Iag(n)) + ...
                %                             spdiags(d2,1,Iag(n),Iag(n)) + ...
                %                             spdiags(d2,-1,Iag(n),Iag(n)) +...
                %                             spdiags(d3,2,Iag(n),Iag(n)) + ...
                %                             spdiags(d3,-2,Iag(n),Iag(n));
                As = ttm(As,Ln,n);
            end
        end
        As = reshape(As.data,[],R);
    else
        As = 0;
    end
    
    A = A .* (Y *X)./...
        (Yh * X+eps + param.normA *A + param.smoothA *As);
    A = max(eps,abs(A));
    
    Yh = A*X';Yh(Weights) = 0;
    
    % Update X
    X = X .* (Y'*A)./(Yh'*A+eps);
    X = max(eps,abs(X));
   
    % Normalize A and X
    MX = max(X);
    X = bsxfun(@rdivide,X,MX);
    A = bsxfun(@times,A,MX);
    Yh = A*X';
    
    % Evaluate relative mean squared error
    rmse(ki) = -20*log10(norm(Y(~Weights) - Yh(~Weights),'fro')/nY);
    rmse_f(ki) = -20*log10(norm(Yh_old(Weights) - Yh(Weights),'fro')/norm(Yh_old,'fro'));
    
    if (ki>2)
        
        fprintf('Iter %d, \t RMSE = %.4d, %.4d, RMSE_f = %.4d, %.4d\n',ki,...
            rmse(ki),abs(rmse(ki) - rmse(ki-1)),rmse_f(ki),abs(rmse_f(ki) - rmse_f(ki-1)));
        
        if (abs(rmse_f(ki) - rmse_f(ki-1)) <= 1e-2*rmse_f(ki-1)) ...
                && ((rmse(ki) > param.maxmse) || (abs(rmse(ki) - rmse(ki-1)) < param.tol))
            break
        end
    end
    Yh_old = Yh;Yh(Weights) = 0;
end
%%

if nargout > 2
    Yh = A*X';
    Yh = kron_folding(Yh,Ix,Ia);
end

A = reshape(A,[Ia,R]);
X = reshape(X,[Ix,R]);
    
end


function param = parseinput(opts)
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','nvec',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('verbose',0);
param.addOptional('maxmse',50);

param.addOptional('normA',0);
param.addOptional('smoothA',0);

param.parse(opts);
param = param.Results;
end

function [U,S,V] = msvd(Y,R)
SzY = size(Y);
if nargin <2
    R = min(SzY);
end

mz = min(SzY); Mz = max(SzY);
if Mz/mz>5
    OPTS = struct('disp',0);
    if SzY(1)>SzY(2)
        
        C = Y'*Y;
        if R == mz
            [V,S] = eig(C);
        else
            [V,S] = eigs(C,R,'LM',OPTS);
        end
        S = (sqrt(diag(S)));
        U = Y*V*diag(1./S);
    else
        C = Y*Y';
        if R == mz
            [U,S] = eig(C);
        else
            [U,S] = eigs(C,R,'LM',OPTS);
        end
        S = (sqrt(diag(S)));
        V = diag(1./S)*U'*Y;
    end
    S = diag(S);
else
    if R < mz
        [U,S,V] = svds(Y,R);
    else
        [U,S,V] = svd(Y);
    end
end
end