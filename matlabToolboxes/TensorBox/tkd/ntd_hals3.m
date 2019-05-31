function [T,output] = ntd_hals3(Y,R,opts)
% HALS algorithms for Nonnegative Tucker Decomposition
% Fast version for 3-way tensor decomposition 
%
% INPUT
% Y: three way tensor or array of size I1 x I2 x I3
% R:  a vector [R1 R2 R3] or scalar if R1 = R2 = R3
% opts: parameter for the decomposition
%   .tol :  tolerence (1e-6)
%   .maxiters        (50)
%   .init:  initialization type: cell array of factors and core tensor, or
%           random or leading singular vectors 
%   .alsinit: ALS (or HOOI)-based initialization
%   
% OUTPUT
% T : t-tensor 
%   T.core: a core tensor of size R1 x R2 x R3
%   T.U{1}, T.U{2}. T.U{3}: three factor matrices of size In x Rn
% fitarr:   array of fits and iterations.
%
% EXAMPLE
%   Y = rand(10,12,13); 
%   R = [3 4 5];
%   opts = ntd_hals3;opts.init = 'nvec';opts.maxiters = 100;
%   [T,fit] = ntd_hals3(Y,R,opts);
%   plot(fit(:,1),1-fit(:,2));xlabel('Iteration');ylabel('Relative Error')
% 
% REF: 	
% Anh Huy Phan, Andrzej Cichocki, Extended HALS algorithm for nonnegative
% Tucker decomposition and its applications for multiway analysis and
% classification, Neurocomputing, vol. 74, 11, pp. 1956-1969, 2011 

% 08/2010
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% April 2013, optimized codes for sparse tensors

if ~exist('opts','var'), opts = struct; end
opts = parseInput(opts);
if nargin == 0
    T = opts; return
end

fprintf('\nHALS-3 NTD:\n');

% Extract number of dimensions and norm of Y.
N = ndims(Y);In = size(Y);


if numel(R) == 1
    R = R(ones(1,N));
end
if numel(opts.lortho) == 1
    opts.lortho = opts.lortho(ones(1,N));
end
% if numel(opts.lda_smooth) == 1
%     opts.lda_smooth = opts.lda_smooth(ones(1,N));
% end

% p_perm = [];
% if ~issorted(In)
%     [In,p_perm] = sort(In);
%     Y = permute(Y,p_perm);
%     R = R(p_perm);
% end

%% Set up and error checking on initial guess for U.
if isa(Y,'double')
    Y = tensor(Y);
end
[A,G] = ntd_init(Y,R,opts);
%[A,G] = ntd_initialize(Y,opts.init,opts.alsinit,R);
AtA = cellfun(@(x)  x'*x, A,'uni',0);fit = inf;

% Convert tensor Y to mode-3 unfolding Y
if isa(Y,'tensor')
    Y = reshape(Y,[prod(In(1:end-1)) In(end)]);
    Y = Y.data;
end
if isa(Y,'sptensor')
    Y = double(sptenmat(Y,N,'t'));
end
normY = norm(Y(:));

if isa(G,'tensor')
    G = G.data;
end
fitarr = [];

if nargout >=2
    output = struct('Uinit',[],'NoIters',[],'Fit',[]);
end

%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(0.001)
    fitold = fit;
    
    % Update Factor
    %Y x3 A3^T
    for n = 1:3 
        if n == 1   % Update A{1} n = 1;
            % common part for updating A{1} and A{2}
            if iter == 1
                YtA3 = Y * A{3};    % I1I2 x R3
                
                GtA = reshape(G,[],R(3)) * AtA{3}; % R1R2 x R3
                GtA = reshape(permute(reshape(GtA,[],R(2),R(3)),[1 3 2]),[],R(2)); % R1R3 x R2
                
                GtA2 = GtA * AtA{2}; %R1 x R3R2
                GtA2 = reshape(GtA2,R(1),[]);
            end
            YtA3 = reshape(permute(reshape(YtA3,[],In(2),R(3)),[1 3 2]),[],In(2)) ; % I1R3 x I2
            
            YtA2 = YtA3* A{2};
            YtA2 = reshape(YtA2,In(1),[]); %I1 x R3R2
            
            G = reshape(permute(G,[1 3 2]),R(1),[]); % R1 x R3R2
            YtAnG = YtA2 * G'; % (Y x_{-1} A^T) x1 G
            B = G * GtA2';
        end
        
        if n== 2 % Update A{2} n = 2;
            YtA2 = A{1}'*reshape(YtA3,In(1),[]); % R1 x R3I2
            YtA2 = reshape(YtA2,[],In(2)); %R1R3 x I2
            
            GtA2 = AtA{1} * reshape(GtA,R(1),[]); %R1 x R3R2
            GtA2 = reshape(GtA2,[],R(2));         %R1R3 x R2
            
            G = reshape(G,[],R(2)); % R1R3 x R2
            YtAnG = (G'*YtA2)'; % I2 x R2 (Y x_{-2} A^T) x2 G
            B = (G' * GtA2)';    % R2 x R2
        end
        
        if n == 3 % Update A{3}
            YtA = A{1}' * reshape(Y,In(1),[]);    % R1 x I2I3
            YtA = reshape(permute(reshape(YtA,[],In(2),In(3)),[2 1 3]),In(2),[]) ; % I2 x R1I3
            YtA = A{2}' * YtA;        % R2 x R1I3
            YtA = reshape(YtA,[],In(3));% R2R1 x I3
            
            GtA = AtA{1} * reshape(G,R(1),[]); % R1x R3R2
            GtA = reshape(GtA,[],R(2)); % R1R3 x R2
            GtA = GtA * AtA{2};         % R1R3 x R2
            GtA = permute(reshape(GtA,R(1),R(3),R(2)),[2 3 1]); %
            GtA = reshape(GtA,R(3),[]); % R3 x R2R1
            
            G = reshape(G,R(1),R(3),R(2));
            G = reshape(permute(G,[3 1 2]),[],R(3)); % R2R1 x R3
            YtAnG = (G'*YtA)'; % I3 x R3
            B = GtA*G;    % R3 x R3
        end
        
        if opts.lortho(n) ~= 0
            As = sum(A{n},2);
        end
        for r = 1:R(n)
            A{n}(:,r) = YtAnG(:,r) - A{n}(:,[1:r-1 r+1:end]) * B([1:r-1 r+1:end],r);
            if opts.lortho(n) ~= 0
                A{n}(:,r) = A{n}(:,r) - opts.lortho(n)*(As -A{n}(:,r));
            end
            A{n}(:,r) = max(1e-10,A{n}(:,r)/B(r,r));
        end
        
        % Normalize factor and correct core tensor
        ellA = sqrt(sum(A{n}.^2,1));%ellA = sqrt(max(A{n},[],1));
        
        if n == 1
            G = bsxfun(@times,G,ellA');
            GtA = bsxfun(@times,reshape(GtA,R(1),[]),ellA'); %R1 x R3R2
        else % n = 2 or n = 3
            G = bsxfun(@times,G,ellA);    %n = 2: R1R3 x R2    %n = 3: R2R1 x R3
        end
        
        A{n} = bsxfun(@rdivide,A{n},ellA);
        AtA{n} = A{n}'*A{n};
    end
    
    % Update core tensor G
    % G = G.*full(ttm(Y,A,'t'))./ttm(G,AtA);    % multiplicative update rule
    G = permute(reshape(G,R(2),R(1),R(3)),[2 1 3]);
    YtA3 = Y*A{3}; %I1*I2 x R3
    for r3 = 1:R(3)
        Yv2 = reshape(YtA3(:,r3),[],In(2)) * A{2}; %I1 x R2
        
        for r2 = 1:R(2)    
            Yv1 = Yv2(:,r2)' * A{1};
            B = AtA{2}(:,r2)*AtA{3}(:,r3)';%B = B(:)';
            GtA2 = reshape(G,R(1),[]) * B(:); %R1 x R2
            %gamma = AtA{2}(r2,r2)*AtA{3}(r3,r3);
            for r1 = 1:R(1)
                gjnew = max(eps, (Yv1(r1) -  AtA{1}(:,r1)'*GtA2)+ G(r1,r2,r3));
                GtA2(r1) = GtA2(r1) + (gjnew- G(r1,r2,r3));
                G(r1,r2,r3) = gjnew;
            end            
        end
    end
    
    % Precompute G x3 AtA{3}
    GtA = reshape(G,[],R(3)) * AtA{3}; % R1R2 x R3
    GtA = reshape(permute(reshape(GtA,[],R(2),R(3)),[1 3 2]),[],R(2)); % R1R3 x R2
    GtA2 = GtA * AtA{2}; %R1 x R3R2
    GtA2 = reshape(GtA2,R(1),[]);
    
    if (mod(iter,opts.printitn)==0) || (iter == opts.maxiters)
        % Compute fit
        % Compute Y x {A'} -> approx. error and innerprod between Y and Yhat = G x {A}
        YtA = YtA * A{3};
        YtA = permute(reshape(YtA,R([2 1 3])),[2 1 3]);
        GtA3 = AtA{1} * GtA2;
        GtA3 = permute(reshape(GtA3,R([1 3 2])),[1 3 2]);
        
        normresidual = sqrt(normY^2 + (GtA3(:) - 2*YtA(:))'*G(:));
        fit = 1 - (normresidual/normY);        %fraction explained by model
        fitchange = abs(fitold - fit);
        fprintf('Iter %2d: fit = %e fitdelta = %7.1e\n', ...
            iter, fit, fitchange);                  % Check for convergence

        fitarr = [fitarr ; iter fit];

        if (fitchange < opts.tol) && (fit>0)
            break;
        end
    end
end
%% Compute the final result
T = ttensor(tensor(G), A);
% if ~isempty(p_perm)
%     T = ipermute(T,p_perm);
% end

if nargout >=2
    output.NoIters = iter;
    output.Fit = fitarr;
end
end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ttensor')||ismember(x(1:4),{'rand' 'nvec' 'fibe' 'nmfs'})));
param.addOptional('maxiters',50);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addParamValue('lsmooth',0);
param.addParamValue('lortho',0);
param.addParamValue('lsparse',0);
param.addParamValue('alsinit',1);

param.parse(opts);
param = param.Results;
end