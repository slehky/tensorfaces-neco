function [Xt,output] = ttmps_a2cu(X,rankX,opts)
% Alternating two-cores update with Left-Right orthogonalization algorithm
% which approximates a tensor X by a TT-tensor of rank rank-X.
%
% Each time of iteration, the algorithm updates two cores simutaneously,
% then updates the next two cores. The update process runs from left to
% right, i.e., from the first core to last core. Then it runs from right to
% left to update cores in the descending order, i.e, N, N-1, ..., 2, 1
%
% In general, the process is as the following order when N is odd
%
%  Left to right : (1,2), (3,4), ..., (N-2,N-1),
%  Right to left : (N,N-1), (N-2, N-3), ..., (3,2)
%  Left to right : (1,2), (3,4), ..., (N-2,N-1)
%
% When N is even, update orders are as follows
%
%  Left to right : (1,2), (3,4), ..., (N-3,N-2),
%  Right to left : (N,N-1), (N-2, N-3), ..., (4,3)
%  Left to right : (1,2), (3,4), ..., (N-3,N-2),

%
% Note, when approximating a tensor X by a TT-tensor with rank specified,
% the algorithm often does not achieve solution after a few rounds of LR-LR
% updates, and may need many iterations. This will require high
% computational cost due to projection of the tensor on to subpsace of
% factor matrices of the estimation tensor.
%
% To this end, instead of fitting the data directly, one should compress
% (fit) X by a TT-tensor with higher accuracy, i.e. higher rank, then
% truncate the TT-tensor to the rank-X using this algorithm.
% This compression procedure is lossless when rank of the approximation
% TT-tensor higher than the specified rank_X.
%
% The algorithm supports compression option as an acceleration method
%
%  Parameters
%
%    compression: 1     1:  prior-compression of X to a TT-tensor. The
%                       algorithm will manipulate on the TT-tensors
%
%    compression_accuracy: 1.0000e-06  :  \|Y - X\|_F <= compression_accuracy
%
%    init: 'nvec'  initialization method
%    maxiters: 200     maximal number of iterations
%
%    noise_level: 1.0000e-06   : standard deviation of noise,
%                     if it is given, ASCU solves the denoising problem,
%                     i.e.,   min  \|Y - X\|_F  <= noise_level^2 * numel(Y)
%
%     normX: []    norm of the tensor X
%
%
%    printitn: 0
%    tol: 1.0000e-06
%
%
%
% Phan Anh Huy, 2016
%


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    Xt = param; return
end


%%
if param.printitn ~=0
    fprintf('\nAlternating Double-Cores Update for TT-decomposition\n');
end

%% Correct ranks of X
N = ndims(X);
if isa(X,'TTeMPS')
    SzX = X.size;
else
    SzX = size(X);
end
if ~isempty(rankX)
    for n = [2:N N-1:-1:2]
        rankX(n) = min([rankX(n) rankX(n-1)*SzX(n-1) rankX(n+1)*SzX(n)]);
    end
end

%% Stage 1: compress the data by a TT-tensor using SVD truncation
maxiters = param.maxiters;

% Compress data to TT-tensor with specific accuracy, not by specified rank
% the compression_accuracy should be at least equal to the noise level.

if param.compression
    if ~isa(X,'tt_tensor')
        if isempty(param.compression_accuracy)
            param.compression_accuracy = param.noise_level;
        end
        X = tt_tensor(X,param.compression_accuracy,SzX,rankX);
        X = TT_to_TTeMPS(X);
        % adjust the norm X in the cost function due to compression
        %normX2 = normX2 + 2*norm(Xtt)^2 - 2 * innerprod(Xtt,X);
        %X = Xtt;
    end
end

%% Precompute norm of the tensor X, which is used for fast assessment of the
% cost function
if ~isempty(param.normX)
    normX2 = param.normX^2;
else
    if isa(X,'tt_tensor') || isa(X,'TTeMPS')
        normX2 = norm(X)^2;
    else
        normX2 = norm(X(:))^2;
    end
end

%% Get initial value or Initialize a TT-tensor by rouding X
Xt = initialization;

%% Output is a tensor orthogonalized from right to left
Xt = orthogonalize(Xt,1);
rankX = Xt.rank; % rank of Xt may change due to the orthogonalization

%% Prepare for the main loop

err = nan(maxiters,1);

prev_error = [];
% stop_  = false;
tol = param.tol;
cnt = 0;

% Core indices in the left-to-right and right-to-left update procedures
% core_step = 1 or 2;
left_to_right_updims = 1:param.core_step:N-param.core_step; %left to right update mode
if mod(N,2) == 0
    right_to_left_updims = [N N-1:-param.core_step:1+param.core_step];  %right to left update mode
else
    right_to_left_updims = N:-param.core_step:1+param.core_step;  %right to left update mode
end

no_updates_L2R = numel(left_to_right_updims);
no_updates_R2L = numel(right_to_left_updims);

% Expand the core index arrays by the first index of the other one.
left_to_right_updims = [left_to_right_updims right_to_left_updims(1)];
right_to_left_updims = [right_to_left_updims left_to_right_updims(1)];

% Precompute left- and right contracted tensors
if isa(X,'TTeMPS')
    % If X is a TT-tensor, the contraction between X and its estimation can
    % be computed cheaper through left and right contraction matrices
    % Phi_left and Phi_right
    rankS = X.rank;
    [Phi_left,Phi_right] = ttmps_contract_init(X,Xt);
elseif isa(X,'tt_tensor')
    rankS = rank(X);
    [Phi_left,Phi_right] = tt_contract_init(X,Xt);
else
    progressive_mode = true; % for progressive computation of contracted tensors Tn
    % progressive_mode = false;
    % progressive_mode = false;
    if progressive_mode
        Phi_left = cell(N,1);
        Phi_left{1} = 1;
    end
end

%% Main part
max_stop_cnt = 6;
stop_cnt = 0;

if ~isempty(param.noise_level)
    % If noise_level is given the algorithm solves the denoising problem
    %     \| Y - X \|_F^2 < noise_level
    % such that X has minimum rank.
    %
    tt_accuracy = param.noise_level;
    tt_accuracy = tt_accuracy - normX2;
end

prev_n = 0; % previous_ core index

for kiter = 1:maxiters
    
    % Rung the left to right update
    % This round will update pair of cores (1,2), (3,4), ...,
    % or (1,2),(2,3),... which depends on the core_step
    
    
    for k_update = 1: no_updates_L2R
        % Core to be updated
        n = left_to_right_updims(k_update);     % core to be updated
        
        % next_n: the next core to be updated
        next_n = left_to_right_updims(k_update+1);
        
        % counter
        cnt = cnt+1;
        
        % Update the core X{n} and X{n+1}
        
        % Truncated SVD of projected data Tn
        % The full objective function is
        %  min  \|Y\|_F^2 - \|Tn\|_F^2 + \| Tn - X{n} * X{n+1} \|_F^2
        %
        % where Tn is left right projection of X by cores of Xt except two
        % cores  n and n+1.
        %
        % which achieves minimum when X{n} * X{n+1} is best rank-(Rn+1)
        % approximation to Tn
        % The objective function is
        %       f_min =  \|Y\|_F^2 - sum(s.^2)
        %
        % where s comprises R_n leading singular values of Tn.
        % modes: arrays of core indices
        modes = [n n+1];
        [u,s,v] = factorize_left_right_proj(modes,prev_n);
        
        % Assess the approximation error - objective function
        %curr_err = (normX2 - norm(Tn,'fro')^2 +  norm(Tn - u*diag(s)*v','fro')^2)/normX2;
        curr_err = (normX2 - sum(s.^2))/normX2;
        
        err(cnt) = curr_err;
        
        if mod(cnt,param.printitn)==0
            fprintf('Iter %d , Cores %12s, Error %.5f\n',kiter,sprintf('%d-',modes),curr_err);
        end
        
        
        % Check convergence
        if (~isempty(prev_error) && (abs(curr_err - prev_error) < tol))
            stop_cnt = stop_cnt+1;
            %break;
        else
            stop_cnt = 0;
        end
        
        prev_error = curr_err;
        
        % Update core tensors X{n} and X{n+1}
        % Update Xn
        Xt.U{n} = reshape(u,rankX(n), SzX(n),[]);
        % Since u is orthogonal, do not need to left-orthogonalize X{n}
        % Update rank of X{n}
        rnp1 = size(u,2); % new rank R(n+1)
        rankX(n+1) = rnp1;
        
        % If the next core to be updated is not X{n+1}, but e.g., X{n+2},
        % Update X{n+1}, and left-orthogonalize this core tensor
        v = v*diag(s);
        
        if next_n >= (n+2) % then next_n == n+2
            % left-Orthogonalization for U{n+1}
            v = reshape(v',rnp1*SzX(n+1),[]);
            [v,vR] = qr(v,0);
            
            rnp2 = size(v,2);
            
            % no need adjust Xt{n+2} because it will be updated in next iteration
            % except the last iteration
%             if k_update == no_updates_L2R
                Xt.U{n+2} = reshape(vR*reshape(Xt.U{n+2},rankX(n+2), []),rnp2,SzX(n+2),[]);
%             end
            
            % Update X{n+1}
            Xt.U{n+1} = reshape(v,rnp1, SzX(n+1),[]);
            
            % Update rank of X{n+1}
            rankX(n+2) = rnp2;
            
        else % if n_next = n+1
            % Update X{n+1}
            %if k_update == no_updates_L2R
            Xt.U{n+1} = reshape(v',rnp1, SzX(n+1),[]);
            %end
        end
          
        prev_n = n;
        
        if stop_cnt > max_stop_cnt
            % If the left-to-right update stops, then
            % update X{n+2}
%             if next_n >= (n+2)
%                 Xt.U{n+2} = reshape(vR*reshape(Xt.U{n+2},rankX(n+2), []),rnp2,SzX(n+2),[]);
%                 %             else
%                 %                 Xt.U{n+1} = reshape(v,rnp1, SzX(n+1),[]);
%             end
            break
        end
        
        
    end
    if stop_cnt > max_stop_cnt
        break
    end
    
    
    %% right to left update
    
    % Last Phi_left which is updated is Phi_left(modes(2))
    for k_update = 1:no_updates_R2L
        n = right_to_left_updims(k_update);        % core to be updated
        next_n = right_to_left_updims(k_update+1); % next core to be updated
        
        cnt = cnt+1;
        
        % Factorization of left and right projection matrix
        %  of size (Rn *In)  x (I(n+1) * R(n+2))
        modes = [n n-1];
        [u,s,v] = factorize_left_right_proj(modes,prev_n);
        
        % assess relative approximation error
        curr_err = (normX2 - sum(s.^2))/normX2;
        err(cnt) = curr_err;
        
        if mod(cnt,param.printitn)==0
            fprintf('Iter %d , Cores %12s, Error %.5f\n',kiter,sprintf('%d-',modes),curr_err);
        end 
        
        if (~isempty(prev_error) && (abs(curr_err - prev_error) < tol)) || (kiter == maxiters)
            stop_cnt = stop_cnt+1;
        else
            stop_cnt = 0;
        end
        prev_error = curr_err;
        
        
        % Update X{n}
        rn = size(v,2);
        Xt.U{n} = reshape(v',rn, SzX(n),rankX(n+1));
        rankX(n) = rn;
        
        % If the next core to be updated is not X{n-1}, but e.g., X{n-2},
        % then Update X{n-1}, and left-orthogonalize this core tensor
        u = u*diag(s);
        
        if next_n <= (n-2)
            
            % Orthogonalization to Un-1 or Un
            % right orthogonalization to U{n-1}
            u = reshape(u,rankX(n-1),[]);
            [u,uR] = qr(u',0);
            
            rnm1 = size(u,2);
            
            % no need adjust Xt{n-2}  because it will be updated in next iteration
%             if k_update == no_updates_R2L
                Xt.U{n-2} = reshape(reshape(Xt.U{n-2},[],rankX(n-1))*uR',rankX(n-2),SzX(n-2),[]);
%             end
            
            % Update X{n-1}
            Xt.U{n-1} = reshape(u',[], SzX(n-1),rn);
            rankX(n-1) = rnm1;
        else
            % X{n-1} even need not to be updated, except the last run
            %if k_update == no_updates_R2L
            Xt.U{n-1} = reshape(u,[], SzX(n-1),rn);
            %end
        end
        
        prev_n = n;
        if stop_cnt > max_stop_cnt
%             if next_n == n-2
%                 Xt.U{n-2} = reshape(reshape(Xt.U{n-2},[],rankX(n-1))*uR',rankX(n-2),SzX(n-2),[]);
                %else
                %    Xt.U{n-1} = reshape(u',[], SzX(n-1),rn);
%             end
            break
        end
    end
    
    if stop_cnt > max_stop_cnt
        break
    end
end

if nargout >=2
    err = err(1:cnt);
    output = struct('Fit',1-err,'NoIters',cnt);
end



%%

    function Xt = initialization
        if isa(param.init,'TTeMPS')
            Xt = param.init;
        elseif isa(param.init,'tt_tensor')
            Xt = param.init;
        else
            % if rankX is not given, noiselevel must be given
            % Truncation of the data
            % tt_eps = sqrt(param.noise_level^2*(N-1));
            tt_eps = 1e-7;
            
            if isempty(rankX)
                Xt = tt_tensor(X,tt_eps,SzX);
                %Xt = tt_stensor(X,1e-8,SzX);
            elseif ~isa(X,'tt_tensor') % TT-Rank is given
                Xt = tt_tensor(X,tt_eps,SzX,rankX);
                Xt = round(Xt,1e-8,rankX);
                %Xt = tt_stensor(X,1e-8,SzX,rankX);
            else
                Xt = round(X,1e-8,rankX);
            end
        end
        if isa(Xt,'tt_tensor')
            Xt = TT_to_TTeMPS(Xt);
        end
        rankX = Xt.rank;
    end


%%

    function [u,s,v] = factorize_left_right_proj(modes,prev_n)
        % Truncated SVD of Tn
        % The full objective function is
        %  min  \|Y\|_F^2 - \|Tn\|_F^2 + \| Tn - G{n} * G{n+1} \|_F^2
        %
        % which achieves minimum when G{n} * G{n+1} is best rank-(Rn+1)
        % approximation to Tn.
        %       f_min =  \|Y\|_F^2 - sum(s.^2)
        %  where s comprises R_n leading singular values of Tn.
        

        modes = sort(modes);
        
        if isa(X,'TTeMPS') %isa(X,'tt_tensor')
            
            % Update left- contracted tensors Phi_left and right-
            % contracted tensors Phi_right
            if prev_n < modes(1)
                % 'LR'
                % Update Phi_left
                for kn3 = 0:modes(1)-prev_n-1
                    if prev_n-1+kn3>=0
                        [Phi_left,Phi_right] = ttmps_contract_update(X,Xt,Phi_left,Phi_right,prev_n-1+kn3,prev_n+kn3);
                    end
                end
                % LEFT to RIGHT
            elseif prev_n > modes(2)
                % 'RL'
                % Update Phi_right
                for kn3 = 0:prev_n-modes(2)-1
                    [Phi_left,Phi_right] = ttmps_contract_update(X,Xt,Phi_left,Phi_right,prev_n-kn3+1,prev_n-kn3);
                end
            end
            
            % modes = sort(modes);
            %Tn = kron(eye(SzX(n)),Phi_left{n}')*reshape(X{n},[],rankS(n+1)) * reshape(X{n+1},rankS(n+1),[])*kron(Phi_right{n+1},eye(SzX(n+1)));
            T_left = reshape(Phi_left{modes(1)}'*reshape(X.U{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
            T_right = reshape(reshape(X.U{modes(1)+1},[],rankS(modes(1)+2))* Phi_right{modes(1)+1},rankS(modes(1)+1),[]);
            
            
            [qlf,rlf] = qr(T_left,0);
            [qrt,rrt] = qr(T_right',0);
            
            Tn = rlf*rrt';
            
        elseif isa(X,'tt_tensor')
            
            % Update left- contracted tensors Phi_left and right-
            % contracted tensors Phi_right
            if prev_n < modes(1)
                % 'LR'
                % Update Phi_left
                for kn3 = 0:modes(1)-prev_n-1
                    if prev_n-1+kn3>=0
                        [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,prev_n-1+kn3,prev_n+kn3);
                    end
                end
                % LEFT to RIGHT
            elseif prev_n > modes(2)
                % 'RL'
                % Update Phi_right
                for kn3 = 0:prev_n-modes(2)-1
                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,prev_n-kn3+1,prev_n-kn3);
                end
            end
            
            
            % modes = sort(modes);
            %Tn = kron(eye(SzX(n)),Phi_left{n}')*reshape(X{n},[],rankS(n+1)) * reshape(X{n+1},rankS(n+1),[])*kron(Phi_right{n+1},eye(SzX(n+1)));
            T_left = reshape(Phi_left{modes(1)}'*reshape(X{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
            T_right = reshape(reshape(X{modes(1)+1},[],rankS(modes(1)+2))* Phi_right{modes(1)+1},rankS(modes(1)+1),[]);
            
            [qlf,rlf] = qr(T_left,0);
            [qrt,rrt] = qr(T_right',0);
            
            Tn = rlf*rrt';
            
        else
            % For a tensor X, the contraction is computed through
            % a progressive computation, in which the left-contraced
            % tensors are saved
            if progressive_mode
                % Update the Phi-Left if needed
                [Phi_left] = contract_update(Xt,X,Phi_left,prev_n,modes(1));
                % Compute the both-side contracted tensor
                [Tn,Phi_left] = contraction_bothsides(modes,Phi_left);
            else
                Tn = ttxt(Xt,X,modes,'both'); % contraction except mode-n
            end
            Tn = reshape(Tn,rankX(modes(1))*SzX(modes(1)),[]);
        end
        
        
        
        %% Factorize Tn
        if isempty(param.noise_level)
            
            [u,s,v] = svd(Tn,0);
            u = u(:,1:rankX(modes(1)+1));
            v = v(:,1:rankX(modes(1)+1));s = diag(s);
            s = s(1:rankX(modes(1)+1));
            
            
        else
            normTn2 = norm(Tn(:))^2;
            
            % When the noise level is given, A2CU solves the denoising
            % problem
            % min \| Y - X\|_F^2 = |Y|_F^2 - |Tn|_F^2 + |T_n-X|_F^2 <= eps
            %
            % i.e.
            %   min   |T_n-X|_F^2  <=  eps_n
            %
            %  where the accuracy level eps_n is given by
            
            accuracy_n = tt_accuracy + normTn2; % eps - |Y|_F^2 + |T_n|_F^2
            if accuracy_n<= 0
                % Negative accuracy_n indicates that the rank is too small
                % to explain the data with the given accuracy error.
                % Rank of the core tensor needs to increase
                
                %Tn = reshape(Tn,rankX(modes(1))*SzX(modes(1)),[]);
                [u,s,v] = svd(Tn,0);
                u = u(:,1:rankX(modes(1)+1));
                v = v(:,1:rankX(modes(1)+1));s = diag(s);
                s = s(1:rankX(modes(1)+1));
                
            else % accuracy_n >  0 % Solve the denoising problem
                %Tn = reshape(Tn,rankX(modes(1))*SzX(modes(1)),[]);
                
                [u,s,v]=svd(Tn,'econ');
                s=diag(s);
                cs = cumsum(s.^2);
                r1 = find((cs(end) - cs) < accuracy_n,1,'first');
                u=u(:,1:r1); s=s(1:r1);v = v(:,1:r1);
                
            end
        end
        
        if isa(X,'TTeMPS') || isa(X,'tt_tensor')
            u = qlf*u;
            v = qrt*v ;
        end
        
    end


    function [Phi_left] = contract_update(Xt,X,Phi_left,n_prev,n_curr)
        % Xt is TT-tensor , X is a tensor of order-N (the same order of Xt
        %
        % Update the left- contraction tensors Phi_left
        %     n_prev:  previous mode
        %     n_curr:  current mode
        
        N = ndims(X);
        % Update Phi_left only when go from left to right 1, 2, ...
        if (n_curr>1)  && (n_prev < n_curr)  % Left-to-Right
            if n_curr == 2
                Z = ttxt(Xt,X,2,'left'); % left contraction of mode-n
                Z = reshape(Z,[rankX(2), SzX(2:end)]);
                Phi_left{2} = Z;
                
            else
                % Z = ttxt(Xt,X,n(1),'left'); % contraction except mode-n
                %  Z = reshape(Z,[rankX(n(1)), SzX(n(1):end)]);
                %
                if n_prev == 1
                    Z = X;
                else
                    Z = Phi_left{n_prev};
                end
                for kn3 = 0:n_curr-n_prev-1
                    Z = reshape(Z,rankX(n_prev+kn3)*SzX(n_prev+kn3),[]);
                    Z = reshape(Xt.U{n_prev+kn3},rankX(n_prev+kn3)*SzX(n_prev+kn3),[])' * Z; % Rn x (In ... IN)
                    
                    Z = reshape(Z,[rankX(n_prev+kn3+1), SzX(n_prev+kn3+1:end)]);
                    Phi_left{n_prev+kn3+1} = Z;
                end
            end
        end
    end

    function [Tn,Phi_left] = contraction_bothsides(n,Phi_left)
        % Compute the contraction matrix between X and Xtt for both-side of
        % modes [n1, n2]
        %
        %  Phi_left{n} is left-contraction matrix between X and Xtt at mode
        %  n
        %
        %  Phi_left{n} can be computed through Phi_left{n-1}
        
        progressive_mode = true;
        if ~progressive_mode
            % Direct computation - expensive method
            Tn = ttxt(Xt,X,n,'both'); % contraction except mode-n
            
        else
            
            if n(1) == 1
                Tn = ttxt(Xt,X,n(2),'right'); % contraction except mode-n
                
            else
                %  Phi-left is precomputed
                Z = Phi_left{n(1)};
                
                % right contraction
                for n2 = N:-1:n(2)+1;
                    if n2 == N
                        Z = reshape(Z,[],SzX(N));
                        Z = Z * Xt.U{n2}'; % R2 x (I2 ... IN)
                    else
                        Z = reshape(Z,[],rankX(n2+1)*SzX(n2)); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = Z *  reshape(Xt.U{n2},[],rankX(n2+1)*SzX(n2))'; % R2 x (I2 ... IN)
                    end
                end
                Tn = reshape(Z,rankX(n(1)),[], rankX(n(2)+1));
            end
        end
    end
end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','nvec',@(x) (iscell(x) || isa(x,'tt_tensor')||...
    isa(x,'TTeMPS') || ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld' 'exac'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('compression',true);
param.addOptional('compression_accuracy',1e-6);
param.addOptional('noise_level',1e-6);% variance of noise
param.addOptional('printitn',0);
param.addOptional('core_step',2,@(x) ismember(x,[1 2])); % or 1: non-overlapping or overlapping update

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;


end

%%
function [Phi_left,Phi_right] = ttmps_contract_init(X,B)

% % Pre-Compute Phi_left % matrices of size R x R, products of all cores form
% % 1:n-1 between two tensor Xtt and Btt
N = ndims(X);
Phi_left = cell(N,1);
Phi_left{1} = 1;
% Since the update is assumed to run from left to right, the
% we don' t need to compute left-contraction matrices Phi_left.

% Pre-Compute Phi_right % matrices of size R x R, products of all cores form
% n+1:N between two tensor Xtt and Btt
Phi_right = cell(N,1);
Phi_right{N} = 1;
Phi_right{N-1} = X.U{N}*B{N}';
% Xe = TT_to_TTeMPS(X); Be = TT_to_TTeMPS(B);
for n = N-2:-1:1
    %dir = 'RL';upto = n+1; % upto = 2, 3, 4
    %Phi_right {n} = innerprod(Xe,Be, 'RL', n+1); % contraction between cores k of X and B where k = n+1:N
    %Phi_right{n} = double(ttt(ttm(tensor(X{n+1}),Phi_right{n+1}',3),tensor(B{n+1}),[2 3]));
    n_curr = n+1;
    tmp  = reshape(X.U{n_curr},[],size(X{n_curr},3)) * Phi_right{n_curr};
    Phi_right{n_curr-1} = reshape(tmp,[],size(B{n_curr},2)*size(B{n_curr},3)) * reshape(B{n_curr},[],size(B{n_curr},2)*size(B{n_curr},3))';
end

end


%%
function [Phi_left,Phi_right] = ttmps_contract_update(X,B,Phi_left,Phi_right,n_prev,n_curr)
% Update the left- and right contraction tensors Phi_left and Phi_right
%     n_prev:  previous mode
%     n_curr:  current mode

N = ndims(X);
% Update Phi_left only when go from left to right 1, 2, ...
if n_prev < n_curr % Left-to-Right
    if n_curr == 1
        Phi_left{n_curr+1} = reshape(X.U{1},X.size(1),[])'*reshape(B.U{1},X.size(1),[]);
    elseif n_curr<N
        %Phi_left{n_curr+1} = double(ttt(ttm(tensor(X{n_curr}),Phi_left{n_curr}',1),tensor(B{n_curr}),[1 2]));
        tmp  = Phi_left{n_curr}' * reshape(X.U{n_curr},X.rank(n_curr),[]);
        Phi_left{n_curr+1} = reshape(tmp,size(B.U{n_curr},1)*size(B.U{n_curr},2),[])' * reshape(B.U{n_curr},size(B.U{n_curr},1)*size(B.U{n_curr},2),[]);
    end
    
else % Update Phi_right only when go from right to left, N,N-1, ...
    if n_curr == N
        Phi_right{n_curr-1} = X.U{N}*B.U{N}';
    elseif n_curr>1
        %Phi_right{n_curr-1} = double(ttt(ttm(tensor(X{n_curr}),Phi_right{n_curr}',3),tensor(B{n_curr}),[2 3]));
        
        tmp  = reshape(X.U{n_curr},[],size(X.U{n_curr},3)) * Phi_right{n_curr};
        Phi_right{n_curr-1} = reshape(tmp,[],size(B.U{n_curr},2)*size(B.U{n_curr},3)) * reshape(B.U{n_curr},[],size(B.U{n_curr},2)*size(B.U{n_curr},3))';
    end
end
end

%%

%%
function [Phi_left,Phi_right] = tt_contract_init(X,B)

% % Pre-Compute Phi_left % matrices of size R x R, products of all cores form
% % 1:n-1 between two tensor Xtt and Btt
N = ndims(X);
Phi_left = cell(N,1);
Phi_left{1} = 1;
% Since the update is assumed to run from left to right, the
% we don' t need to compute left-contraction matrices Phi_left.

% Pre-Compute Phi_right % matrices of size R x R, products of all cores form
% n+1:N between two tensor Xtt and Btt
Phi_right = cell(N,1);
Phi_right{N} = 1;
Phi_right{N-1} = X{N}*B{N}';
% Xe = TT_to_TTeMPS(X); Be = TT_to_TTeMPS(B);
for n = N-2:-1:1
    %dir = 'RL';upto = n+1; % upto = 2, 3, 4
    %Phi_right {n} = innerprod(Xe,Be, 'RL', n+1); % contraction between cores k of X and B where k = n+1:N
    %Phi_right{n} = double(ttt(ttm(tensor(X{n+1}),Phi_right{n+1}',3),tensor(B{n+1}),[2 3]));
    n_curr = n+1;
    tmp  = reshape(X{n_curr},[],size(X{n_curr},3)) * Phi_right{n_curr};
    Phi_right{n_curr-1} = reshape(tmp,[],size(B{n_curr},2)*size(B{n_curr},3)) * reshape(B{n_curr},[],size(B{n_curr},2)*size(B{n_curr},3))';
end

end

%%
function [Phi_left,Phi_right] = tt_contract_update(X,B,Phi_left,Phi_right,n_prev,n_curr)
% Update the left- and right contraction tensors Phi_left and Phi_right
%     n_prev:  previous mode
%     n_curr:  current mode

N = ndims(X);
% Update Phi_left only when go from left to right 1, 2, ...
if n_prev < n_curr % Left-to-Right
    if n_curr == 1
        Phi_left{n_curr+1} = reshape(X{1},size(X,1),[])'*reshape(B{1},size(X,1),[]);
    elseif n_curr<N
        %Phi_left{n_curr+1} = double(ttt(ttm(tensor(X{n_curr}),Phi_left{n_curr}',1),tensor(B{n_curr}),[1 2]));
        tmp  = Phi_left{n_curr}' * reshape(X{n_curr},size(X{n_curr},1),[]);
        Phi_left{n_curr+1} = reshape(tmp,size(B{n_curr},1)*size(B{n_curr},2),[])' * reshape(B{n_curr},size(B{n_curr},1)*size(B{n_curr},2),[]);
    end
    
else % Update Phi_right only when go from right to left, N,N-1, ...
    if n_curr == N
        Phi_right{n_curr-1} = X{N}*B{N}';
    elseif n_curr>1
        %Phi_right{n_curr-1} = double(ttt(ttm(tensor(X{n_curr}),Phi_right{n_curr}',3),tensor(B{n_curr}),[2 3]));
        
        tmp  = reshape(X{n_curr},[],size(X{n_curr},3)) * Phi_right{n_curr};
        Phi_right{n_curr-1} = reshape(tmp,[],size(B{n_curr},2)*size(B{n_curr},3)) * reshape(B{n_curr},[],size(B{n_curr},2)*size(B{n_curr},3))';
    end
end
end