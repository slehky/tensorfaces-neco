function [Xt,output] = ttmps_a3cu(X,rankX,opts)
% Alternating two-cores update with Left-Right orthogonalization algorithm
% which approximates a tensor X by a TT-tensor of rank rank-X.
%
% Each time of iteration, the algorithm updates three cores simutaneously,
% then updates the next three cores. The update process runs from left to
% right, i.e., from the first core to last core. Then it runs from right to
% left to update cores in the descending order, i.e, N, N-1, ..., 2, 1
%
% In general, the process is as the following order
%
%  Left to right : (1,2,3), (3,4,5), ..., (N-5,N-4,N-3),
%  Right to left : (N-2,N-1,N), ..., (4,5,6)
%  Left to right : (1,2,3), (3,4,5), ..., (N-3,N-2,N-1),
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
% Phan Anh Huy, 2016
%


%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    Xt = param; return
end

if param.printitn ~=0
    fprintf('\nAlternating Triple-Cores Update for TT-decomposition\n');
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
    if isa(X,'TTeMPS') || isa(X,'tt_tensor')
        normX2 = norm(X)^2;
    else
        normX2 = norm(X(:))^2;
    end
end

%% Get initial value or Initialize a TT-tensor by rouding X
Xt = initialization;

%% Output is a tensor orthogonalized from right to left
Xt = orthogonalize(Xt,1);
% rankX = rank(Xt); % rank of Xt may change due to the orthogonalization
rankX = Xt.rank;

%% Prepare for the main loop

err = nan(maxiters,1);

prev_error = [];
% stop_  = false;
tol = param.tol;
cnt = 0;

left_ix = 1;
right_ix = 1;
  
left_to_right_updims = left_ix:param.core_step:N-2; %left to right update mode
%right_to_left_updims = N-right_ix+1:-param.core_step:3;  %right to left update mode
right_to_left_updims = [N N-right_ix:-param.core_step:3];  %right to left update mode

no_updates_L2R = numel(left_to_right_updims);
no_updates_R2L = numel(right_to_left_updims);


%     % Expand the core index arrays by the first index of the other one.
left_to_right_updims = [left_to_right_updims right_to_left_updims(1)];
right_to_left_updims = [right_to_left_updims left_to_right_updims(1)];


% Precompute contract tensors Phi_left and Phi_right

if isa(X,'TTeMPS')
    % If X is a TT-tensor, the contraction between X and its estimation can
    % be computed cheaper through left and right contraction matrices
    % Phi_left and Phi_right
    %rankS = rank(X);
    rankS = X.rank;
    %[Phi_left,Phi_right] = tt_contract_init(X,Xt);
    [Phi_left,Phi_right] = ttmps_contract_init(X,Xt);

elseif isa(X,'tt_tensor')
    % If X is a TT-tensor, the contraction between X and its estimation can
    % be computed cheaper through left and right contraction matrices
    % Phi_left and Phi_right
    rankS = rank(X);
    [Phi_left,Phi_right] = tt_contract_init(X,Xt);
else
    progressive_mode = true; % for progressive computation of contracted tensors Tn
    % progressive_mode = false;
    %     progressive_mode = false;
    if progressive_mode
        Phi_left = cell(N,1);
    end
end


%% Main part
max_stop_cnt = 4;
stop_cnt = 0;


if ~isempty(param.noise_level)
    % If noise_level is given the algorithm solves the denoising problem
    %     \| Y - X \|_F^2 < noise_level
    % such that X have minimum rank.
    %
    tt_accuracy = param.noise_level;
    tt_accuracy = tt_accuracy - normX2;
end
prev_n = 0;

for kiter = 1:maxiters
    
    % left to right update
    % This round will update pair of cores (1,2,3), (4,5,6), ...
    %     left_ix = mod(left_ix,2)+1;
%     %     right_ix = mod(right_ix,2)+1;
%     
%     left_to_right_updims = left_ix:param.core_step:N-2; %left to right update mode
%     right_to_left_updims = N-right_ix+1:-param.core_step:3;  %right to left update mode
%     
%     no_updates_L2R = numel(left_to_right_updims);
%     no_updates_R2L = numel(right_to_left_updims);
%     
%     
% %     % Expand the core index arrays by the first index of the other one.
%     left_to_right_updims = [left_to_right_updims right_to_left_updims(1)];
%     right_to_left_updims = [right_to_left_updims left_to_right_updims(1)];
%     
    
    
    for k_update = 1: no_updates_L2R
        n = left_to_right_updims(k_update);
        % next_n: the next core to be updated
        next_n = left_to_right_updims(k_update+1);
        
        cnt = cnt+1;
        
        % Solve a Tucker-2 decomposition of a projected data
        %       (G_{n+1} x_1 [G_{n}]_{(3)}^T  x_3 [G_{n+2}]_{(1)}^T
        modes = [n n+1 n+2];
 
        [G1,G2,G3] = factorize_left_right_proj(modes,prev_n,param.exacterrorbound);
        
        % G1 is a new estimate of X{n},
        % G2 is a new estimate of X{n+1},
        % and G3 is a new estimate of X{n+2},
        
        % Assess the approximation error
        curr_err = (normX2 - norm(G2(:))^2)/normX2;
        err(cnt) = curr_err;
       
        if mod(cnt,param.printitn)==0
            fprintf('Iter %d , Cores %12s, Error %.5f\n',kiter,sprintf('%d-',modes),curr_err);
        end
        
        % Check convergence
        if (~isempty(prev_error) && (abs(curr_err - prev_error) < tol))
            stop_cnt = stop_cnt+1;
        else
            stop_cnt = 0;
        end
        
        prev_error = curr_err;
        
        % Left-Orthogonalization modes(1), modes(2), modes(3)
        %         for km = modes(2:3)
        %             Xt = orth_at(Xt, km, 'left' );
        %         end
        
        
        % Since G1 is left-orthogonal, we need not orthogonalize X{modes{1})
        % If the next core to be updated is not X{n+1}, but e.g.,
        % X{n+2},...
        % Update X{n+1}, and left-orthogonalize this core tensor
        
        if next_n >= (n+2) 
            % Then orthogonalize X{n+1}, i.e. G2,
            
            % Orthogonalize G2, i.e. X{n+1}
            % Since slices of G2 are orthogonal upto scaling,
            %    unfold(G2,'left')'*unfold(G2,'left') = diag(lambda)
            % left orthogonalization of G2 is simply scaling frontal slices of G2, by its norm.
            % Left-Orthogonalize G2
            lambdaG2 = sqrt((sum(sum(G2.^2,1),2))); % this is also singular vector when computing G3
            G2 = bsxfun(@rdivide,G2,lambdaG2);
            G3 = bsxfun(@times,G3,lambdaG2(:));
        end
        
        if next_n >= (n+3)
             % Then orthogonalize X{n+2}, i.e. G3,
             % Orthogonalize G3
             if modes(3)<N
                 G3 = reshape(G3,[],rankX(modes(3)+1));
                 [G3,rr] = qr(G3,0);
                 G3 = reshape(G3,[],SzX(modes(3)),size(rr,1));
                 G4 = rr*reshape(Xt.U{modes(3)+1},rankX(modes(3)+1),[]);
                 G4 = reshape(G4,[],SzX(modes(3)+1),rankX(modes(3)+2));
                 Xt.U{modes(3)+1} = G4; rankX(modes(3)+1) = size(rr,1);
             end
        end
        
        Xt.U{modes(1)} = G1;rankX(modes(2)) = size(G2,1);
        Xt.U{modes(2)} = G2;
        Xt.U{modes(3)} = G3;rankX(modes(2)+1) = size(G2,3);
        
        % Left-Orthogonalize G3
        %Xt = orth_at(Xt, modes(3), 'left' );
        %rankX(modes(3)+1) = Xt.rank(modes(3)+1);%size(Xt{modes(3)},3);        
        
         
        prev_n = n;
        
        if stop_cnt > max_stop_cnt
            break
        end
        
    end
    
    if stop_cnt > max_stop_cnt
        break
    end
    
    %% right to left update
    
    if ~isempty(right_to_left_updims) && (right_to_left_updims(1) < N)
        Xt = orthogonalize_upto(Xt,right_to_left_updims(1),'right');
        rankX(right_to_left_updims(1)+1:end) = Xt.rank(right_to_left_updims(1)+1:N+1);
    end
    
    % The last Phi_left which has been updated is Phi_left(modes(2))
    for k_update = 1:no_updates_R2L
        n = right_to_left_updims(k_update);
        next_n = right_to_left_updims(k_update+1);
        
        cnt = cnt+1;
        
        % Factorization of left and right projection tensor
        %  of size (Rn *In)  x I(n+1) x (I(n+2) * R(n+3))
        modes = [n n-1 n-2];
        [G1,G2,G3] = factorize_left_right_proj(modes,prev_n,param.exacterrorbound);
        
        % G3 is a new estimate of X{n},
        % G2 is a new estimate of X{n-1},
        % and G1 is a new estimate of X{n-2},
        
        % assess relative approximation error
        curr_err = (normX2 - norm(G2(:))^2)/normX2;
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
        
        % Since G3 is right-orthogonal, we need not orthogonalize X{modes{3})
        
        % Right-Orthogonalize G2
        if next_n <= (n-2)
            % Then Right-Orthogonalize G2
            % Since slices of G2 are orthogonal upto scaling,
            %    unfold(G2,'right')*unfold(G2,'right')' = diag(lambda)
            % Right orthogonalization of G2 is simply scaling frontal slices of G2, by its norm.
            
            lambdaG2 = sqrt((sum(sum(G2.^2,3),2))); % this is also singular vector when computing G3
            G2 = bsxfun(@rdivide,G2,lambdaG2);
            G1 = bsxfun(@times,G1,reshape(lambdaG2(:),1,1,[]));
        end
        
        % Right orthogonalize G1
        if next_n <= (n-3)
            if modes(3)>1
                G1 = reshape(G1,rankX(modes(3)),[]);
                [G1,rr] = qr(G1',0);
                G1 = reshape(G1',size(rr,1),SzX(modes(3)),[]);
                
                G0 = reshape(Xt.U{modes(3)-1},[],rankX(modes(3))) * rr';
                G0 = reshape(G0,rankX(modes(3)-1),SzX(modes(3)-1),[]);
                Xt.U{modes(3)-1} = G0; rankX(modes(3)) = size(rr,1);
            end
        end
        
        Xt.U{modes(3)} = G1;rankX(modes(3)+1) = size(G2,1);
        Xt.U{modes(2)} = G2;rankX(modes(2)+1) = size(G2,3);
        Xt.U{modes(1)} = G3;        
        
        prev_n = n;
        if stop_cnt > max_stop_cnt
            break
        end
    end
    
    if stop_cnt > max_stop_cnt
        break
    end
end

if nargout >=2
    err = err(1:cnt,:);
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

    function [G1,G2,G3] = factorize_left_right_proj(modes,prev_n,exacterrorbound)
        % Tucker -2 decomposition to estimate three core G2 x1 G1 x3 G3
        % to the projection of tensor X onto subspace of the rest (N-3)
        % core tensors Gk, k \notin modes
        
        if modes(1)<modes(end)
            dirupdate = 'LR';
        else
            dirupdate = 'RL';
        end
        modes = sort(modes);
        
        compress_1 = false;compress_3 = false;
        % Previous values of core tensors X{mode_1} and X{mode_3}
        G1 = reshape(Xt.U{modes(1)},rankX(modes(1))*SzX(modes(1)),[]);
        G3 = reshape(Xt.U{modes(1)+2},[],rankX(modes(1)+3)*SzX(modes(1)+2))';
        
        % Compute the contraction tensor 
        if isa(X,'TTeMPS')
            
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
            
            %Pb = kron(eye(SzX(n)),Phi_left{n}')*reshape(X{n},[],rankS(n+1)) * reshape(X{n+1},rankS(n+1),[])*kron(Phi_right{n+1},eye(SzX(n+1)));
            T_left = reshape(Phi_left{modes(1)}'*reshape(X.U{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
            T_right = reshape(reshape(X.U{modes(1)+2},[],rankS(modes(1)+3))* Phi_right{modes(1)+2},rankS(modes(1)+2),[]);
            T_right = T_right';
            % Phi_left{modes(1)} :
            compress_1 = size(T_left,1)>size(T_left,2);
            compress_3 = size(T_right,1)<size(T_right,2);
            
            if compress_1 % compress the factor
                [qlf,T_left] = qr(T_left,0);
            end
            if compress_3 % compress the factor
                [qrt,T_right] = qr(T_right,0);
            end
            
            % Tucker tensor
            %Pb = ttm(tensor(X{modes(1)+1}),{T_left T_right},[1 3]);
            Pb = T_left * reshape(X.U{modes(1)+1},size(T_left,2),[]);
            Pb = reshape(Pb,[],size(T_right,2)) * T_right';
            Pb = reshape(Pb,size(T_left,1),[],size(T_right,1));
            
            G1 = reshape(Xt.U{modes(1)},rankX(modes(1))*SzX(modes(1)),[]);
            if compress_1 % compress the factor
                G1 = qlf'*G1;
            end
            G3 = (reshape(Xt.U{modes(1)+2},[],rankX(modes(1)+3)*SzX(modes(1)+2))');
            if compress_3
                G3 = qrt'*G3;
            end
             
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
            
            %Pb = kron(eye(SzX(n)),Phi_left{n}')*reshape(X{n},[],rankS(n+1)) * reshape(X{n+1},rankS(n+1),[])*kron(Phi_right{n+1},eye(SzX(n+1)));
            T_left = reshape(Phi_left{modes(1)}'*reshape(X{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
            T_right = reshape(reshape(X{modes(1)+2},[],rankS(modes(1)+3))* Phi_right{modes(1)+2},rankS(modes(1)+2),[]);
            T_right = T_right';
            % Phi_left{modes(1)} :
            compress_1 = size(T_left,1)>size(T_left,2);
            compress_3 = size(T_right,1)<size(T_right,2);
            
            if compress_1 % compress the factor
                [qlf,T_left] = qr(T_left,0);
            end
            if compress_3 % compress the factor
                [qrt,T_right] = qr(T_right,0);
            end
            
            % Tucker tensor
            %Pb = ttm(tensor(X{modes(1)+1}),{T_left T_right},[1 3]);
            Pb = T_left * reshape(X{modes(1)+1},size(T_left,2),[]);
            Pb = reshape(Pb,[],size(T_right,2)) * T_right';
            Pb = reshape(Pb,size(T_left,1),[],size(T_right,1));
            
            if compress_1 % compress the factor
                G1 = qlf'*G1;
            end
            if compress_3
                G3 = qrt'*G3;
            end
            
             
        else % if X is a tensor
            
            % For a tensor X, the contraction is computed through
            % a progressive computation, in which the left-contraced
            % tensors are saved
            if progressive_mode %&& strcmp(dirupdate,'LR')
                % Update the Phi-Left if needed
                [Phi_left] = contract_update(Xt,X,Phi_left,prev_n,modes(1));
                % Compute the both-side contracted tensor
                [Pb,Phi_left] = contraction_bothsides(modes,Phi_left,dirupdate);
            else
                Pb = ttxt(Xt,X,modes,'both'); % contraction except mode-n
            end
            
            % reshape to an order-3 tensor
            Pb = reshape(Pb,rankX(modes(1))*SzX(modes(1)),SzX(modes(1)+1),[]);
            
        end
        
        % Orthogonalize G1 and G3
        [G1,~] = qr(G1,0);
        [G3,~] = qr(G3,0);
        
        %% Compress the order-3 tensor Pb (J1 x J2 x J3)
        % if  J1 > (J2*J3)   or (J1*J2) < J3
        szPb = size(Pb);
        compressPb = 0;
        if (szPb(1) * szPb(2)) <  szPb(3)
            compressPb = 3;
            
            % compress the 3-rd mode
            Pb = reshape(Pb,[],szPb(3));
            [Pb_proj,Pb] = qr(Pb',0);
            Pb = reshape(Pb',szPb(1), szPb(2),[]);
            G3 = Pb_proj'*G3;
            
           
            
        elseif szPb(1) >  (szPb(2)*szPb(3))
            compressPb = 1;
            % compress the 1-st mode
            Pb = reshape(Pb,szPb(1),[]);
            [Pb_proj,Pb] = qr(Pb,0); 
            Pb = reshape(Pb,[],szPb(2), szPb(3));
            G1 = Pb_proj'*G1;
        end
        
        %% Factorize data 
        % Fit this tensor by a Tucker-2 decomposition
        %     Pb ~~ G2 x1 [G1]_(1,2) x3 [G3]_(1)^T
             
        if isempty(param.noise_level)
            [G1,G2,G3] = fast_tucker2(Pb,G1,G3,10,1e-8);
        else
            normTn2 = norm(Pb(:))^2;
            
            % When the noise level is given, A2CU solves the denoising
            % problem
            % min \| Y - X\|_F^2 = |Y|_F^2 - |Tn|_F^2 + |T_n-X|_F^2 <= eps
            %
            % i.e.
            %   min   |T_n-X|_F^2  <=  eps_n
            %
            %  where the accuracy level eps_n is given by
            
            accuracy_n = tt_accuracy + normTn2; % eps - |Y|_F^2 + |T_n|_F^2
            
            if accuracy_n < 0
                % If accuracy is negative, the rank is small and should be
                % increased
                
                % [G1,~] = qr(G1,0);
                [G1,G2,G3] = fast_tucker2(Pb,G1,G3,10,1e-8);
                
                
            else % accuracy_n >  0 % Solve the denoising problem
                %Pb = reshape(Pb,rankX(modes(1))*SzX(modes(1)),[],rankX(modes(1)+3)*SzX(modes(1)+2));
                %[G1,G2,G3] = fast_tucker2_denoising(Pb,'nvecs',100,1e-4,sqrt(accuracy_n/numel(Pb)));
                  
                Gi = {G1 G3};
                %[G1,G2,G3] = fast_tucker2_denoising(Pb,Gi,100,1e-4,sqrt(accuracy_n/numel(Pb)));
                [G1,G2,G3] = fast_tucker2_denoising(Pb,'nvecs',100,1e-4,sqrt(accuracy_n/numel(Pb)),exacterrorbound);
                %[G1,G2,G3] = fast_tucker2_denoising(Pb,Gi,100,1e-4,sqrt(accuracy_n/numel(Pb)),exacterrorbound);
            end 
        end
        
        %% 
        switch compressPb 
            case 1
                G1 = Pb_proj*G1;
            case 3
                G3 = Pb_proj*G3;
        end
        
        
        %%
        
        if compress_1
            G1 = qlf*G1 ;
        end
        if compress_3
            G3 = (qrt*G3) ;
        end
         
          
        G1 = reshape(G1,rankX(modes(1)),SzX(modes(1)),[]);
        G3 = reshape(G3',[],SzX(modes(1)+2),rankX(modes(1)+3));
        
    end


    function [u,s,v] = factorize_left_right_proj_2(modes)
        % Truncated SVD of Tn
        % The full objective function is
        %  min  \|Y\|_F^2 - \|Tn\|_F^2 + \| Tn - G{n} * G{n+1} \|_F^2
        %
        % which achieves minimum when G{n} * G{n+1} is best rank-(Rn+1)
        % approximation to Tn.
        %       f_min =  \|Y\|_F^2 - sum(s.^2)
        %  where s comprises R_n leading singular values of Tn.
        
        if modes(1)<modes(2)
            dirupdate = 'LR';
        else
            dirupdate = 'RL';
        end
        modes = sort(modes);
        
        if isa(X,'tt_tensor')
            % modes = sort(modes);
            %Tn = kron(eye(SzX(n)),Phi_left{n}')*reshape(X{n},[],rankS(n+1)) * reshape(X{n+1},rankS(n+1),[])*kron(Phi_right{n+1},eye(SzX(n+1)));
            T_left = reshape(Phi_left{modes(1)}'*reshape(X{modes(1)},rankS(modes(1)),[]),[],rankS(modes(1)+1));
            T_right = reshape(reshape(X{modes(1)+1},[],rankS(modes(1)+2))* Phi_right{modes(1)+1},rankS(modes(1)+1),[]);
            
            [qlf,rlf] = qr(T_left,0);
            [qrt,rrt] = qr(T_right',0);
            
            % truncated SVD to Tn
            [u,s,v] = svd(rlf*rrt',0);
            u = qlf*u(:,1:rankX(modes(1)+1));
            v = qrt*v(:,1:rankX(modes(1)+1));s = diag(s);
            s = s(1:rankX(modes(1)+1));
            
        else
            % For a tensor X, the contraction is computed through
            % a progressive computation, in which the left-contraced
            % tensors are saved
            if progressive_mode
                [Tn,Phi_left] = contraction_bothsides(modes,Phi_left,dirupdate);
            else
                Tn = ttxt(Xt,X,modes,'both'); % contraction except mode-n
            end
            
            
            Tn = reshape(Tn,rankX(modes(1))*SzX(modes(1)),[]);
            [u,s,v] = svd(Tn,0);
            u = u(:,1:rankX(modes(1)+1));
            v = v(:,1:rankX(modes(1)+1));s = diag(s);
            s = s(1:rankX(modes(1)+1));
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


    function [Tn,Phi_left] = contraction_bothsides(n,Phi_left,dirupdate)
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
                Tn = ttxt(Xt,X,n(end),'right'); % contraction except mode-n
                
            else
                if strcmp(dirupdate,'LR')
                    
                    %  % Compute Phileft{n(1)}
                    %  Z = ttxt(Xt,X,n(1),'left'); % contraction except mode-n
                    %  Z = reshape(Z,[rankX(n(1)), SzX(n(1):end)]);
                    %  Phi_left{n(1)} = Z;
                    
                    if n(1)==2
                        Z = ttxt(Xt,X,2,'left'); % left contraction of mode-n
                        Z = reshape(Z,[rankX(2), SzX(2:end)]);
                        Phi_left{2} = Z;
                        
                    else
                        Z = Phi_left{n(1)-1};
                        nk = n(1)-1;
                        for kn = 0 %0:n(2)-n(1)
                            Z = reshape(Z,rankX(nk+kn)*SzX(nk+kn),[]); % In x (In+1... IN) R2 R3 ...R(n-1)
                            Z = reshape(Xt{nk+kn},rankX(nk+kn)*SzX(nk+kn),[])' * Z; % R2 x (I2 ... IN)
                            
                            Z = reshape(Z,[rankX(nk+kn+1), SzX(nk+kn+1:end)]);
                            Phi_left{nk+kn+1} = Z;
                        end
                    end
                    
                else
                    Z = Phi_left{n(1)};
                end
                
                % right contraction
                for n2 = N:-1:n(end)+1;
                    if n2 == N
                        Z = reshape(Z,[],SzX(N));
                        Z = Z * Xt{n2}'; % R2 x (I2 ... IN)
                    else
                        Z = reshape(Z,[],rankX(n2+1)*SzX(n2)); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = Z *  reshape(Xt{n2},[],rankX(n2+1)*SzX(n2))'; % R2 x (I2 ... IN)
                    end
                end
                Tn = reshape(Z,rankX(n(1)),[], rankX(n(end)+1));
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
param.addOptional('core_step',2,@(x) ismember(x,[1:3])); % or 1: non-overlapping or overlapping update
param.addOptional('exacterrorbound',true);% |Y-Yx|_F = noise level

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

%%


function [U1,U2,U3] = fast_tucker2(X,U1,U3,maxiters,tol)
% Simple fast Tucker-2 decomposition
% Tk = mtucker_als(Pb,[rankX(modes(1)+1) rankX(modes(1)+2)],tk_opts);
SzX = size(X);
X1 = reshape(X,size(X,1),[]);
X3 = reshape(X,[],size(X,3));
r1 = size(U1,2);
if r1>SzX(1)
    [U1,~] = qr(U1,0);
    r1 = size(U1,2);
end

r3 = size(U3,2);
if r3>SzX(3)
    [U3,~] = qr(U3,0);
    r3 = size(U3,2);
end
normU2 = zeros(maxiters,1);

% The loop updates U1 and U3, while U2 is computed after the iteration
for kiter = 1:maxiters
    % Update U3
    T = U1'*X1;
    T = reshape(T,[],SzX(3));
    Q = T'*T;
    
    [u,e] = eig(Q);
    [e,id] = sort(diag(e),'descend');
    U3 = u(:,id(1:r3)); % X1 * (I ox U1*U1') X1'
    
    % Update U1
    T = X3*U3;
    T = reshape(T,SzX(1),[]);
    Q = T*T';
    
    [u,e] = eig(Q);
    [e,id] = sort(diag(e),'descend');
    U1 = u(:,id(1:r1));
    
    % Stop if converged
    normU2(kiter) = sum(e(1:r1));
    % err(kiter) = norm(X - U1 x U2 x U3)^2 
    %            = norm(X)^2 - norm(U2)^2
    if (kiter>1) && (abs(normU2(kiter) - normU2(kiter-1))<= tol)
        break
    end
end
U2 = reshape(U1'*T,r1,[],r3);
end