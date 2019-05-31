function [A,Err] = mprocrustes(X,Z,opts)
%
% Procrustes problem
% Given X of size I x J
% Find orthogonal A{1} , ..., A{P} of size I x Rp such that 
% min \|X - Xh\|_F
% where Xh = A1 * Z1 + ... + AP * ZP
%
% Phan Anh Huy, 2013
% TENSORBOX

% Example : 
% R = [4 5 6]; P = numel(R);
% I = sum(R); J = I;
% 
% Y = 0;
% A0 = cell(P,1);
% Z = cell(P,1);
% for p = 1:P
%     A0{p} = orth(rand(I,R(p)));
%     Z{p} = randn(R(p),J);
%     Y = Y + A0{p} * Z{p};
% end
% 
% A = mprocrustes(Y,Z);

%%
if ~exist('opts','var'), opts = struct; end

param = inputParser;
param.KeepUnmatched = true;
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('verbose',false);
param.addOptional('normX',[]);
param.addOptional('init',[]);
param.parse(opts);
param = param.Results;
if nargin == 0;
    A = param; return
end
if isempty(param.normX)
    param.normX = norm(X,'fro');
end


R = cellfun(@(x) size(x,1),Z,'uni',1);
P = numel(Z);

% Initialization
if isempty(param.init)
    A = cell(P,1);
    Xh = 0;
    for p = 1:P
        if p ~=P
            [u,foe,v] = svds(X*Z{p}',R(p));
        else
            [u,foe,v] = svds((X-Xh)*Z{p}',R(p));
        end
        A{p} = u * v';
        Xh = Xh + A{p} * Z{p};
    end
elseif iscell(param.init)
    A = param.init;
    Xh = 0;
    for p = 1:P
        Xh = Xh + A{p} * Z{p};
    end
end

%%
Err = [];errold = inf;
done = false;
while ~done
    
    for p = 1:P
        Xmp = Xh - A{p} * Z{p};
        Xr = X - Xmp;
        [u,foe,v] = svds(Xr*Z{p}',R(p));
        
        A{p} = u * v';
        Xh = Xmp + A{p} * Z{p};
    end
    
    
    err = norm(X - Xh,'fro')/param.normX;
    Err = [Err err];
    if param.verbose
        fprintf('Err %d \n',err);
    end
    if ~isinf(errold)
        done = abs(err - errold)<param.tol;
    end
    errold = err;
end
% semilogy(Err)