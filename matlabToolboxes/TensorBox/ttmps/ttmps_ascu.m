function [Xt,output] = tt_ascu(X,rankX,opts)
% Alternating single core update with Left-Right orthogonalization algorithm
% which approximates a tensor X by a TT-tensor of rank rank-X.
%
% Note, instead of fitting the data directly a rank-X TT-tensor using this
% implementation, one should compress (fit) X by a TT-tensor with higher accuracy,
% i.e. higher rank, than truncate the TT-tensor to rank-X
%
%   tol = 1e-5;
%   Xt1 = tt_stensor(X,tol,SzX,rankX);
%
%   opts = tt_ascu;
%   [Xt,output] = tt_ascu(Xt1,rankX,opts);
%
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
%    rankadjust: 1 or 2
%                     number of modes to be adjusted : 1- left or right side, 2-both sides
%                     Set rankadjust to 1 in order to update rank of only one side of Xn
%                     rankadjust = 1;
%                     Set rankadjust to 2 in order to update ranks of both sides of Xn
%                     rankadjust = 2;
% 
%    printitn: 0
%    tol: 1.0000e-06
%
% Phan Anh Huy, 2016
%


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    Xt = param; return
end
 
% number of modes to be adjusted : 1- left or right side, 2-both sides
% Set rankadjust to 1 in order to update rank of only one side of Xn
% rankadjust = 1;
% Set rankadjust to 2 in order to update ranks of both sides of Xn
% rankadjust = 2;
rankadjust = param.rankadjust;


%%
if param.printitn ~=0
    fprintf('\nAlternating Single Core Update Algorithm for TT decomposition\n');
end

%% Correct ranks of X
N = ndims(X);
if isa(X,'TTeMPS')
    SzX = X.size;
else
    SzX = size(X);
end

if ~isempty(rankX)
    for n = 2:N
        rankX(n) = min([rankX(n) rankX(n-1)*SzX(n-1) rankX(n+1)*SzX(n)]);
    end
    for n = 2:N
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

%% Initialization 
Xt = initialization;

%% Orthogonalization of Xt 
% Xt = TTeMPS_to_TT(orthogonalize(TT_to_TTeMPS(Xt),1));
Xt = orthogonalize(Xt,1);
rankX = Xt.rank; % rank of Xt may change due to the orthogonalization

%% Precompute Left- and Right contracted tensors 

if isa(X,'tt_tensor')
    % If X is a TT-tensor, the contraction between X and its estimation can
    % be computed cheaper through left and right contraction matrices
    % Phi_left and Phi_right
    rankS = rank(X); % rank of the input TT-tensor
    [Phi_left,Phi_right] = tt_contract_init(X,Xt);
elseif isa(X,'TTeMPS')
    % If X is a TT-tensor, the contraction between X and its estimation can
    % be computed cheaper through left and right contraction matrices
    % Phi_left and Phi_right
    rankS = X.rank; % rank of the input TT-tensor
    [Phi_left,Phi_right] = ttmps_contract_init(X,Xt);
else
    progressive_mode = true; % for progressive computation of contracted tensors Tn
                             % progressive_mode = false;
    if progressive_mode 
        Phi_left = cell(N,1);
    end
end

%%
err = nan(maxiters,1);
prev_error = [];
stop_  = false;
tol = param.tol;
cnt = 0;
max_stop_cnt = 12;
stop_cnt = 0;
                                        

if ~isempty(param.noise_level)
    % If noise_level is given the algorithm solves the denoising problem
    %     \| Y - X \|_F^2 < noise_level
    % such that X have minimum rank.
    %
    tt_accuracy = param.noise_level;
    tt_accuracy = tt_accuracy - normX2;
end

for kiter = 1:maxiters
    
    for udir = [1 2]
        % Switch between Left-Right and Right-Left update procedures         
        if udir == 1     % left to rightexae
            update_dims = 1:N-1;dirupdate = 'LR';
        else                 % right to left
            update_dims = N:-1:2; dirupdate = 'RL';
        end
        
        for n = update_dims
            cnt = cnt+1;
            modes = n; % core to be updated
            
            % Tensor Contraction between data X and (N-1) cores of Xt
            % yields a tensor of size :  rn x In x r(n+1)
            
            if isa(X,'tt_tensor')
                % For a TT-tensor X, the contraction is computed through
                % the left and righ contraction matrices Phi_left and
                % Phi_right
                Tn = reshape(Phi_left{modes(1)}'*reshape(X{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
                Tn = Tn * Phi_right{modes(1)};
                
            elseif isa(X,'TTeMPS')
                % For a TT-tensor X, the contraction is computed through
                % the left and righ contraction matrices Phi_left and
                % Phi_right
                Tn = reshape(Phi_left{modes(1)}'*reshape(X.U{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
                Tn = Tn * Phi_right{modes(1)};
                
            else
                % For a tensor X, the contraction is computed through
                % a progressive computation, in which the left-contracted
                % tensors are saved 
                if progressive_mode 
                    [Tn,Phi_left] = contraction_all_but_one(modes(1),Phi_left,dirupdate);
                else
                    Tn = ttxt(Xt,X,modes(1),'both'); % contraction except mode-n
                end
            end
            
         
            % Update the core Xn by best TT-approx to the contracted tensor
            % Tn.
            % 
            %
            normTn2 = norm(Tn(:))^2;
            
            if isempty(param.noise_level)
                % When there is no noise constraint, i.e.  min \| Tn - Xn\|_F^2
                % the new estimate Xn is the contracted tensor Tn
                
                Xt.U{n} = reshape(Tn,rankX(n), SzX(n),rankX(n+1));
                approx_error_Tn = 0;
                orthogonalization(udir);
                                 
            else
                % When the noise level is given, ASCU solves the denoising
                % problem
                % min \| Y - X\|_F^2 = |Y|_F^2 - |Tn|_F^2 + |T_n-X|_F^2 <= eps
                %
                % i.e. 
                %   min   |T_n-X|_F^2  <=  eps_n
                % 
                %  where the accuracy level eps_n is given by 
                
                accuracy_n = tt_accuracy + normTn2; % eps - |Y|_F^2 + |T_n|_F^2
                
                if accuracy_n<= 0
                    % When the accuracy level is negative, Xn is Tn.
                    Xt.U{n} = reshape(Tn,rankX(n), SzX(n),rankX(n+1));
                    approx_error_Tn = 0;
                    orthogonalization(udir);
                    
                   
                else % accuracy_n >  0 % Solve the denoising problem 
                    
                    if (rankX(n)>1) && (rankX(n+1)>1) && (rankadjust == 2)
                        Tn = reshape(Tn,rankX(n),[],rankX(n+1));
                        %[Uleft,Xn,Uright] = fast_tucker2_denoising(Tn,'nvecs',100,1e-4,sqrt(accuracy_n/numel(Tn)));
                        [Uleft,Xn,Uright] = fast_tucker2_denoising(Tn,'nvecs',100,1e-4,sqrt(accuracy_n/numel(Tn)),param.exacterrorbound);

                        %T1 = ttm(tensor(Xn),{Uleft Uright},[1 3]);
                        
                        approx_error_Tn = normTn2 - norm(Xn(:))^2; % |T_n-X|_F^2
                        r1 = size(Uleft,2);r2 = size(Uright,2);
                         
                        % Prepare for orthogonalization 
                        switch udir
                            case 1 % left orthogonalization
                                % Since slices of Xn are orthogonal upto scaling,
                                %    unfold(Xn,'left')'*unfold(Xn,'left') = diag(lambda)
                                % left orthogonalization of Xn is simply scaling frontal slices of Xn, by its norm.
                                % Left-Orthogonalize Xn
                                lambdaG2 = sqrt((sum(sum(Xn.^2,1),2))); % this is also singular vector when computing G3
                                Xn = bsxfun(@rdivide,Xn,lambdaG2);
                                Uright = bsxfun(@times,Uright,lambdaG2(:)');
                                 
                            case 2 % right orthogonalization
                                % Since slices of Xn are orthogonal upto scaling,
                                %    unfold(Xn,'right')*unfold(Xn,'right')' = diag(lambda)
                                % Right orthogonalization of G2 is simply scaling frontal slices of G2, by its norm.
                                % Right-Orthogonalize G2
                                lambdaG2 = sqrt((sum(sum(Xn.^2,3),2))); % this is also singular vector when computing G3
                                Xn = bsxfun(@rdivide,Xn,lambdaG2);
                                Uleft = bsxfun(@times,Uleft,lambdaG2(:)');
                        end
                        
                        % check if Uleft is an identity matrix
                        if (size(Uleft,1) == r1)
                            signU = sign(diag(Uleft));
                            Uleft = bsxfun(@times,Uleft,signU');
                            Xn = bsxfun(@times,Xn,signU(:));
                        end 
        
                        % adjust the contraction matrices Phi_left and
                        % Phi_right 
                        if isa(X,'tt_tensor')
                            %
                            % Phi_left = Phi_left * Uleft;
                            % Phi_right = Uright'*Phi_right;
                            Phi_left{n} = Phi_left{n} * Uleft;
                            Phi_right{n} = Phi_right{n} * Uright;
                            
                        elseif udir == 1
                            % adjust the contraction matrices Phi_left when
                            % running the left-to-right update
                            % becauses the X{n-1} is updated
                            szp = size(Phi_left{n});
                            Phi_left{n} = reshape(Uleft'*reshape(Phi_left{n},szp(1),[]),[r1 szp(2:end)]);
                        end
                        
                        core1 = reshape(reshape(Xt.U{n-1},[],rankX(n)) * Uleft,[],SzX(n-1),r1);
                        core2 = reshape(Uright'*reshape(Xt.U{n+1},rankX(n+1),[]),r2,SzX(n+1),[]);
                        
                        Xt.U{n-1} = core1;
                        Xt.U{n} = Xn;
                        Xt.U{n+1} = core2;
                        rankX(n) = r1;
                        rankX(n+1) = r2;
                         
                         
                    else % Update rank only one side of Xn
                         
                        switch udir
                            case 1 % LR
                                Tn = reshape(Tn,[],rankX(n+1));
                            case 2 % RL
                                Tn = reshape(Tn,rankX(n),[]);
                        end
                        
                        % low-rank approximation to Tn with a bound accuracy_n
                        % |T - Tx|_F^2 <= accuracy_n
                        [u,s,v,approx_error_Tn] = lowrank_matrix_approx(Tn,accuracy_n,param.exacterrorbound);
                        r1 = size(u,2);
%                         [u,s,v]=svd(Tn,'econ');
%                         s=diag(s);
%                         cs = cumsum(s.^2);
%                         r1 = find((cs(end) - cs) <= accuracy_n,1,'first');
%                         approx_error_Tn = cs(end)-cs(r1);
%                         u=u(:,1:r1); s=s(1:r1);v = v(:,1:r1);
                        
                        switch udir
                            case 1 % update from left-to-right
                                
                                core1 = reshape(u,rankX(n),SzX(n),[]);
                                core2 = reshape(diag(s)*v'*reshape(Xt.U{n+1},rankX(n+1),[]),r1,SzX(n+1),[]);
                                
                                Xt.U{n} = core1;
                                
                                % this is no need to update X{n+1} because
                                % it will be estimated in next iteration
                                Xt.U{n+1} = core2;
                                rankX(n+1) = r1;
                                
                                % Update the left- and right contracted tensors
                                if isa(X,'TTeMPS')
                                    [Phi_left,Phi_right] = ttmps_contract_update(X,Xt,Phi_left,Phi_right,n-1,n);
                                elseif isa(X,'tt_tensor')
                                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,n-1,n);
                                end
                                
                                
                            case 2 % update from right-to-left
                                
                                core2 = reshape(v',[],SzX(n),rankX(n+1));
                                core1 = reshape(reshape(Xt.U{n-1},[],rankX(n)) * u*diag(s),[],SzX(n-1),r1);
                                
                                % this is no need to update X{n-1} because
                                % it will be estimated in next iteration
                                Xt.U{n-1} = core1;
                                Xt.U{n} = core2;
                                rankX(n) = r1;
                                
                                if isa(X,'TTeMPS')
                                    [Phi_left,Phi_right] = ttmps_contract_update(X,Xt,Phi_left,Phi_right,n+1,n);
                                elseif isa(X,'tt_tensor')
                                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,n+1,n);
                                    % elseif progressive_mode
                                    % No need to update the left contracted tensor
                                    % Phi_left{n} when the algorithm runs the right-to-left
                                    % update
                                end 
                        end
                        
                        % Orthogonalization the core Xn
                        %orthogonalization(udir);
                        % No need to orthogonalize core tensors
                    end
                end
            end
                 
            % assess error
            curr_err = (normX2 - normTn2 + approx_error_Tn)/normX2;
             
            err(cnt) = curr_err;
             
            if mod(cnt,param.printitn) == 0 
                fprintf('Iter %d , Cores %12s, Error %.5f\n',kiter,sprintf('%d-',n),curr_err);
            end
            
            if ((kiter > 1) && ~isempty(prev_error) && (abs(curr_err - prev_error) < tol)) %|| (kiter == maxiters)
                stop_cnt = stop_cnt+1;
            else
                stop_cnt = 0;
            end
            prev_error = curr_err; 
            
        end
        if stop_cnt > max_stop_cnt
            break
        end
    end 
    if stop_cnt > max_stop_cnt
        break
    end
    if cnt > maxiters
        break
    end
end
 
if nargout >=2
    err = err(1:cnt,:);
    output = struct('Fit',1-err,'NoIters',cnt);
end 

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

    function [Tn,Phi_left] = contraction_all_but_one(n,Phi_left,dirupdate)
        progressive_mode = true;
        if progressive_mode
            
            if n == 1
                Tn = ttxt(Xt,X,n,'right'); % contraction except mode-n
                
            else
                if strcmp(dirupdate,'LR')
                    if n == 2
                        Z = reshape(X,SzX(1),[]);
                        Z = reshape(Xt.U{n-1},SzX(1),[])' * Z; % R2 x (I2 ... IN)
                    else
                        Z = Phi_left{n-1};
                        Z = reshape(Z,rankX(n-1)*SzX(n-1),[]); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = reshape(Xt.U{n-1},rankX(n-1)*SzX(n-1),[])' * Z; % R2 x (I2 ... IN)
                    end
                    Z = reshape(Z,[rankX(n), SzX(n:end)]);
                    Phi_left{n} = Z;
                else
                    % except when n = N, Phi_left{N} needs to be updated
                    if n == N
                        Z = Phi_left{n-1};
                        Z = reshape(Z,rankX(n-1)*SzX(n-1),[]); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = reshape(Xt.U{n-1},rankX(n-1)*SzX(n-1),[])' * Z; % R2 x (I2 ... IN)
                        Z = reshape(Z,[rankX(n), SzX(n:end)]);
                        Phi_left{n} = Z;
                    end
                    Z = Phi_left{n};
                end
                
                % right contraction
                for n2 = N:-1:n+1;
                    if n2 == N
                        Z = reshape(Z,[],SzX(N));
                        Z = Z * Xt.U{n2}'; % R2 x (I2 ... IN)
                    else
                        Z = reshape(Z,[],rankX(n2+1)*SzX(n2)); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = Z *  reshape(Xt.U{n2},[],rankX(n2+1)*SzX(n2))'; % R2 x (I2 ... IN)
                    end
                end
                Tn = reshape(Z,[rankX(n) SzX(n) rankX(n2)]);
            end
            
        else
            Tn = ttxt(Xt,X,n,'both'); % contraction except mode-n
        end
    end


    function orthogonalization(udir)
        % Orthogonalization of Xn and update contracted tensors Phi_left
        % and Phi_right
        switch udir
            case 1 % left orthogonalization
                [Qn,Rn] = qr(reshape(Xt.U{n},[],rankX(n+1)),0);
                
                Xn = reshape(Qn,rankX(n), SzX(n),[]);
                
                % There may not need to adjust the next core (n+1),
                % because it is will be updated in the next iteration.
                % The below step can be done if X{n+1} is not updated
                Xnp1 = reshape(Rn* reshape(Xt.U{n+1},rankX(n+1),[]),[],SzX(n+1),rankX(n+2));
                
                Xt.U{n} = Xn;
                Xt.U{n+1} = Xnp1;
                
                % update rank of X(n+1) if Rn is a fat matrix
                rankX(n+1) = size(Rn,1);
                
                % Update the left- and right contracted tensors
                if isa(X,'TTeMPS')
                    [Phi_left,Phi_right] = ttmps_contract_update(X,Xt,Phi_left,Phi_right,n-1,n);
                elseif isa(X,'tt_tensor')
                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,n-1,n);
                end
                
            case 2 % right orthogonalization
                [Qn,Rn] = qr(reshape(Xt.U{n},rankX(n),[])',0);
                %
                Xn = reshape(Qn',[], SzX(n),rankX(n+1));
                
                % The below step can be done if X{n-1} is not updated
                Xnm1 = reshape(reshape(Xt.U{n-1},[],rankX(n))*Rn',rankX(n-1),SzX(n-1),[]);
                
                
                Xt.U{n} = Xn;
                Xt.U{n-1} = Xnm1;
                
                % Update rank X(n) if Qn is a fat matrix
                rankX(n) = size(Rn,1);
                
                if isa(X,'TTeMPS')
                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,n+1,n);
                    %elseif progressive_mode
                    % No need to update the left contracted tensor
                    % Phi_left{n} when the algorithm runs the right-to-left
                    % update
                elseif isa(X,'tt_tensor')
                    [Phi_left,Phi_right] = tt_contract_update(X,Xt,Phi_left,Phi_right,n+1,n);
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
param.addOptional('noise_level',1e-6);    % |Y-Yx|_F <= noise level
param.addOptional('exacterrorbound',true);% |Y-Yx|_F = noise level

param.addOptional('printitn',0);
param.addOptional('normX',[]);

% number of modes to be adjusted : 1- left or right side, 2-both sides
% Set rankadjust to 1 in order to update rank of only one side of Xn
% rankadjust = 1;
% Set rankadjust to 2 in order to update ranks of both sides of Xn
% rankadjust = 2;
param.addOptional('rankadjust',1); % 

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


function [Phi_left,Phi_right] = tt_contract_update(X,B,Phi_left,Phi_right,n_prev,n_curr)
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
 