function [Ah,Xh,Yh,Yhp,A,X,rmse] = ktc_nng_mgrp(Yex,Ix,P,opts)
% Multiple group Kronecker tensor decomposition with nonnegativity constraint.
%
% The input tensor Y is approximated as
%
%     Y = A_1 \ox  X_1 + ... + A_R \ox  X_R
%
% where \ox denotes the Kronecker product between two tensors A_r and X_r.
% Tensors X_r in the same group g have the same size Ix(g,1) x Ix(g,2) x ...
% x Ix(g,N).
%
% For single group decomposition, see ktc_nng_sgrp.m.
%
% Nan respresents missing entry in Y.
%
% Input:
% Input:
%   Y   :  data tensor with nan respresenting missing entries in Y.
%   Ix  :  G x N array, each row specifies size of tensors X_r in the same
%          group. There are G groups of patterns
%   P   :  vector of length G x 1 indicates the number of tensors X_r in
%          groups.
%   opts:  parameters of the decomposition.
%          Run the algorithm without any input to get the default parameter
%          of the algorithm:
%          opts = ktc_nng;
%
% Output
%   Ah:    cell array whose each entry is of order-(N+1) tensor comprising
%          tensors A_r in the same group, e.g.,
%             Ah{1}(:,:,2) is the second tensor in the first group.
%   Xh:    cell array comprises tensors X_r in the same group.
%
%   Yh  :  approximate tensor
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
    Ah = param; return
end

% for missing data
Weights = isnan(Yex);
Yex(Weights) = 0;

Iy = size(Yex); %N = ndims(Yex);
Ia = bsxfun(@rdivide,Iy,Ix);
 
NoGroups = numel(P);

if numel(param.normA) == 1
    param.normA = param.normA(ones(NoGroups,1));
end

if numel(param.smoothA) == 1
    param.smoothA = param.smoothA(ones(NoGroups,1));
end


RY = cell(NoGroups,1);
A = cell(NoGroups,1);
X = cell(NoGroups,1);
Yhp = cell(NoGroups,1);

% Initialization
Yh = zeros(Iy);

if iscell(param.init)
    if ~isempty(param.init{1})
        A = param.init{1};
        X = param.init{2};
    else
        param.init = 'nvec';
    end
end
for g = 1: NoGroups
    RY{g} = kron_unfolding(Yex,Ix(g,:),Ia(g,:));
    
    if ischar(param.init) && strcmp(param.init(1:4),'nvec')
        temp = min([size(RY{g}) P(g)]);
        [Ag,Sg,Xg] = msvd(RY{g},temp);
        Ag =  Ag*Sg;
        if P(g) > temp
            Ag(:,end+1:P(g)) = rand(size(RY{g},1),P(g)-temp);
            Ag(:,end+1:P(g)) = rand(size(RY{g},2),P(g)-temp);
        end
        A{g} =  abs(Ag); X{g} = abs(Xg);
    elseif ischar(param.init) && strcmp(param.init(1:4),'rand')
        Ag =  rand(size(RY{g},1),P(g));
        Xg =  rand(size(RY{g},2),P(g));
        A{g} =  (Ag); X{g} = (Xg);
    end
    RYp = A{g}*X{g}';
    Ypp = kron_folding(RYp,Ix(g,:),Ia(g,:));
    
    Yh = Yh + Ypp;
%     RYhp{g} = RYp;
    Yhp{g} = Ypp;
end
nY = norm(Yex(:),'fro');

Yh = 0;
for g = 1:NoGroups    
    A{g} = abs(A{g}); X{g} = abs(X{g});
    Ypp = kron_folding(A{g}*X{g}',Ix(g,:),Ia(g,:));
    
    Ypp(Weights) = 0;
    
    Yh = Yh +Ypp;
    % RYhp{g} = RYp;
    Yhp{g} = Ypp;
end


%%
Wg = cell(NoGroups,1);
for g = 1:NoGroups
    Wg{g} = kron_unfolding(Weights,Ix(g,:),Ia(g,:));
end

%% Iterate to extract nonnegative bases
rmse = -20*log10(norm(Yex(:) - Yh(:),'fro')/nY);
for ki  = 1:param.maxiters
    currmse = rmse(end);
    for g1 = 1:max(1,NoGroups-1)
        for g2 = min(NoGroups,g1+1):NoGroups
            gk = unique([g1 g2]);
            rmse_g = rmse(end);
            for ki2 = 1:param.maxiters
                for kg = 1:numel(gk)% 1:NoGroups
                    g = gk(kg);
                    
                    % Kronecker tensor unfolding
                    RYht = kron_unfolding(Yh,Ix(g,:),Ia(g,:));
                    
                    RYp = A{g}*X{g}';
                    RYp(Wg{g}) = 0;
                    
                    % Smoothness constraints on A
                    if param.smoothA  ~= 0
                        Iag = Ia(g,:);
                        Ag = reshape(A{g},[Ia(g,:),P(g)]);
                        Ag = tensor(Ag);
                        for n = 1:numel(Iag)
                            if Iag(n)> 2
                                d1 = [8 9 6*ones(1,Iag(n)-3) 5]';
                                d2 = [-8 -4*ones(1,Iag(n)-2)]';
                                d3 = [2 ones(1,Iag(n)-3)]';
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
                                Ag = ttm(Ag,Ln,n);
                            end
                        end
                        Ag = reshape(Ag.data,[],P(g));
                    else
                        Ag = 0;
                    end
                    
                    A{g} = A{g} .* (RY{g} *X{g})./...
                        (RYht * X{g}+eps + param.normA(g) *A{g} + param.smoothA(g) *Ag);
                    
                    A{g} = max(eps,abs(A{g}));
                    
                    RYht = RYht - RYp;
                    RYp = A{g}*X{g}';RYp(Wg{g}) = 0;
                    RYht = RYht + RYp;
                    
                    X{g} = X{g} .* (RY{g}'*A{g})./(RYht'*A{g}+eps);
                    X{g} = max(eps,abs(X{g}));
                    
 
                    RYp = A{g}*X{g}';RYp(Wg{g}) = 0;
                    
                    % Kronecker tensor folding
                    Ypp = kron_folding(RYp,Ix(g,:),Ia(g,:));

                    Yh = Yh - Yhp{g} +Ypp;
                    Yhp{g} = Ypp;
                    
                    % Normalization
                    MX = max(X{g});
                    X{g} = bsxfun(@rdivide,X{g},MX);
                    A{g} = bsxfun(@times,A{g},MX);
                end
                
                
                rmse_g(ki2) = -20*log10(norm(Yex(:) - Yh(:),'fro')/nY);
                if (ki2>2)
                    fprintf('%d-%d \t %d \t %d\n',g1,g2,rmse_g(ki2),abs(rmse_g(ki2) - rmse_g(ki2-1)));
                end
                if (rmse_g(ki2) > param.maxmse)
                    break
                end
                
                if (ki2>2) && (abs(rmse_g(ki2) - rmse_g(ki2-1)) < param.tol)
                    break
                end
            end
            rmse = [rmse rmse_g];
            if (rmse(end) > param.maxmse)
                break
            end
            
            %             % Visualize
            %             Yhf = 0;
            %             for g = 1: NoGroups
            %                 RYp = A{g}*X{g}';
            %                 Ypp = reshape(RYp,[Ia(g,:) Ix(g,:)]);
            %                 Ypp = ipermute(Ypp,[2:2:2*N 1:2:2*N]);
            %                 Ypp = reshape(Ypp,Iy);
            %                 Yhf = Yhf +Ypp;
            %             end
            %set(hld,'cdata',max(0,min(1,Yhf(:,:,:,13)))); drawnow
        end
    end
    if (rmse(end) > param.maxmse)
        break
    end
    if (abs(rmse(end) - currmse) < param.tol)
        break
    end
end

%%
clear Xh Ah
Yh = 0;
for g = 1: NoGroups
    Ah{g} = reshape(A{g},[Ia(g,:),P(g)]);
    Xh{g} = reshape(X{g},[Ix(g,:),P(g)]);
    %     figure(g);clf
    %     visual(X{g},Ix(g,1),ceil(sqrt(P(g))))
    
    Ypp = kron_folding(A{g}*X{g}',Ix(g,:),Ia(g,:));
    
    Yh = Yh +Ypp;
    Yhp{g} = Ypp;
end

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


function Ym = kron_unfolding(Y,Ix,Ia)
if nargin<3
    Ia = size(Y)./Ix;
end
N = ndims(Y);
Ym = reshape(Y,reshape([Ix; Ia],[],1)');
Ym = permute(Ym,[2:2:2*N 1:2:2*N]);
Ym = reshape(Ym,prod(Ia), prod(Ix));
end

function Y = kron_folding(Ym,Ix,Ia)
N = numel(Ix);
Y = reshape(Ym,[Ia Ix]);
Y = ipermute(Y,[2:2:2*N 1:2:2*N]);
Y = reshape(Y,Ix.*Ia);
end