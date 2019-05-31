function [T,fitarr] = ntd_QALS(Y,R,opts)
% ALS algorithms for NTF based on Nonnegative Quadratic Programming
% Copyright by Phan Anh Huy, 08/2010
% Ref: Novel Alternating Least Squares Algorithm for
%      Nonnegative Matrix and Tensor Factorizations
%      Anh Huy Phan, Andrzej Cichocki, Rafal Zdunek, and Thanh Vu Dinh
%      ICONIP 2010

defoptions = struct('tol',1e-6,'maxiters',50,'init','random',...
    'ellnorm',2,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,...
    'fixsign',0,'rowopt',1);
if ~exist('opts','var')
    opts = struct;
end
opts = scanparam(defoptions,opts);

% Extract number of dimensions and norm of Y.
N = ndims(Y);
normY = norm(Y);

if numel(R) == 1
    R = R(ones(1,N));
end
if numel(opts.lda_ortho) == 1
    opts.lda_ortho = opts.lda_ortho(ones(1,N));
end
if numel(opts.lda_smooth) == 1
    opts.lda_smooth = opts.lda_smooth(ones(1,N));
end
In = size(Y);
%% Set up and error checking on initial guess for U.
[A,G] = ntd_initialize_classify(Y,opts.init,opts.orthoforce,R);
A{N} = speye(In(N));
G = tensor(G);
%%
fprintf('\nLocal NTD:\n');
% Compute approximate of Y
AtA = cellfun(@(x)  x'*x, A,'uni',0);fit = inf;
fitarr = [];

%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(0.001)
    fitold = fit;
    
    % Update Factor
    %YtA = ttm(Y,A,N-1,'t');
%     Ym = reshape(Y,[],In(N-1),In(N));
%     Ym = reshape(permute(Ym,[1 3 2]),[],In(N-1));
%     Ym = Ym * A{N-1}; % (1:(N-2))N x R_(N-1)
    
    for n = 1: N-1
        
        YtA = ttm(Y,A,-[n N],'t');
        YtAn = tenmat(YtA,n);
        
%         YtA = reshape(Ym,prod(In(1:n-1)),In(n),[]);
%         YtA = YtA * A{n}
%         p
        
        for kii = 1:1
            Gn = tenmat(G,n);
            YtAnG = YtAn * Gn';
            
            GtA = full(ttm(G,AtA,-[n N])); % here slow step 
            GtAn = tenmat(GtA,n);
            B = Gn * GtAn';
            if opts.lda_ortho(n) ~= 0
                As = sum(A{n},2);
            end
            for r = 1:R(n)
                A{n}(:,r) = YtAnG(:,r) ...
                    - A{n}(:,[1:r-1 r+1:end]) * B([1:r-1 r+1:end],r);
                if opts.lda_ortho(n) ~= 0
                    A{n}(:,r) = A{n}(:,r) - opts.lda_ortho(n)*(As -A{n}(:,r));
                end
                A{n}(:,r) = max(1e-10,A{n}(:,r)/B(r,r));
            end
            ellA = sqrt(sum(A{n}.^2,1));
            G = ttm(G,diag(ellA),n);
            A{n} = bsxfun(@rdivide,A{n},ellA);
            AtA{n} = A{n}'*A{n};
        end
    end
    %     G = G.*full(ttm(Y,A,'t'))./ttm(G,AtA);   % Frobenius norm
    
%     for r3 = 1:R(3)
%         va = A{3}(:,r3);
%         Ava{3} = AtA{3}(:,r3);
%         ava(3)= AtA{3}(r3,r3);
%         Yv3 = ttv(Y,va,3);
%         
%         for r2 = 1:R(2)
%             va = A{2}(:,r2);
%             Ava{2} = AtA{2}(:,r2);
%             ava(2)= AtA{2}(r2,r2);
%             Yv2 = ttv(Yv3,va,2);
%             
%             for kii = 1: 1
%                 for r1 = 1:R(1)
%                     va = A{1}(:,r1);
%                     Ava{1} = AtA{1}(:,r1);
%                     ava(1)= AtA{1}(r1,r1);
%                     gjnew = max(eps, ttv(Yv2,va,1) - ttv(G,Ava) + G(r1,r2,r3));
%                     G(r1,r2,r3) = gjnew;
%                 end
%             end
%         end
%     end

%     Y4 = reshape(Y.data,[],In(4));
    %G = reshape(G.data,[],In(4));
    
    %% PARFOR
%     IN = In(4);
%     G = G.data;
%     parfor r4 = 1:IN
%         Yv3 = reshape(Y.data(:,:,:,r4),[],In(3)) *A{3}; %I1*I2 x R3
%         G3 = G(:,:,:,r4); 
%         for r3 = 1:R(3)
%             Yv2 = reshape(Yv3(:,r3),[],In(2)) * A{2}; %I1 x R2
%             
%             for r2 = 1:R(2)
%                 Yv1 = Yv2(:,r2)' * A{1};
%                 B = AtA{2}(:,r2)*AtA{3}(:,r3)';%B = B(:)';
%                 GtA2 = reshape(G3,R(1),[]) * B(:); %R1 x R2
%                 gamma = AtA{2}(r2,r2)*AtA{3}(r3,r3);
%                 for r1 = 1:R(1)
%                     gjnew = max(eps, Yv1(r1) -  AtA{1}(:,r1)'*GtA2 + G3(r1,r2,r3));
%                     GtA2(r1) = GtA2(r1) + (gjnew- G3(r1,r2,r3))* gamma;
%                     G3(r1,r2,r3) = gjnew;
%                 end
%             end
%         end
%         G(:,:,:,r4) = G3;
%     end

    %%
    G = G.data;
    for r4 = 1:In(4)
        Yv3 = reshape(Y.data(:,:,:,r4),[],In(3)) *A{3}; %I1*I2 x R3
        %G3 = G(:,:,:,r4);
        for r3 = 1:R(3)
            Yv2 = reshape(Yv3(:,r3),[],In(2)) * A{2}; %I1 x R2
            
            for r2 = 1:R(2)
                Yv1 = Yv2(:,r2)' * A{1};
                B = AtA{2}(:,r2)*AtA{3}(:,r3)';%B = B(:)';
                GtA2 = reshape(G(:,:,:,r4),R(1),[]) * B(:); %R1 x R2
                gamma = AtA{2}(r2,r2)*AtA{3}(r3,r3);
                for r1 = 1:R(1)
                    gjnew = max(eps, Yv1(r1) -  AtA{1}(:,r1)'*GtA2 + G(r1,r2,r3,r4));
                    GtA2(r1) = GtA2(r1) + (gjnew- G(r1,r2,r3,r4))* gamma;
                    G(r1,r2,r3,r4) = gjnew;
                end
            end
        end
    end
    %%
    G = tensor(G);
    Yhat = ttensor(G,A);
    if (mod(iter,5) ==1) || (iter == opts.maxiters)
        % Compute fit
        normresidual = sqrt(normY^2 + norm(Yhat)^2 -2*innerprod(Y,Yhat));
        fit = 1 - (normresidual/normY);        %fraction explained by model
        fitchange = abs(fitold - fit);
        fprintf('Iter %2d: fit = %e fitdelta = %7.1e\n', ...
            iter, fit, fitchange);                  % Check for convergence
        if (fitchange < opts.tol) && (fit>0)
            break;
        end
        fitarr = [fitarr fit];
    end
end
%% Compute the final result
T = ttensor(G, A);
end