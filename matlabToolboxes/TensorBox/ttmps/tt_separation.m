function [xk,err,Ah,ytt] = tt_separation(Y,R,rank_x,opts)
% Separation of a tensor Y into a sum of R TT-tensors x_1,x_2,...,x_R
% whose rank are given in rank-x
%
%  Y = a_1 X_1 + ... + a_R X_R
%
% OR
%
%  Y1 = a_11 X_1 + ... + a_1R X_R
%   ...
%  Ym = a_m1 X_1 + ... + a_mR X_R
%
%  where Y is concatenated of all Y1, ..., Ym
%  i.e., Y(:,...,:,1) = Y1, ...
%        Y(:,...,:,2) = Y2.
%
% Phan Anh-Huy, 2016
%
% R is the number of TT-tensor components 
%
% rank_x: array indicating the TT-rank of each term x{r}, r = 1,..,R
%       rank_x(r,:)   rank of x{r}
%

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    xk = param; return
end


% Set the first and last rank of x to 1s
rank_x(1) = 1;
rank_x(end) = 1;
if isvector(rank_x)
    % All TT-tensors have the same rank
    rank_x = repmat(rank_x(:)',R,1);
end

szy = size(Y);
ndims_y = ndims(Y);
ndims_x = size(rank_x,2)-1;

no_mixtures = 1;
if ndims_x < ndims_y % we consider the cases ndims_y = ndims_x +1, and ndims_y = ndims_x
    % the last dimension of Y indicates the number mixtures, i.e. number of
    % rows of the mixing matrix A: m x R
    % In this case, rank of X{r} is expanded to 1
    no_mixtures = szy(end);
    rank_x(:,ndims_y) = 1;
end

%% Fit a TT-tensor to the tensor Y, if Y is not a TT-tensor.
rank_y = [];
TT_tol = param.tol;
if ~isa(Y,'tt_tensor') && ~strcmp(param.tt_fit_type,'none')
    if isempty(rank_y)
        rank_y = sum(rank_x);
        rank_y(:,1) = 1;
    end
    
    switch param.tt_fit_type(1:5)
        case 'trunc' %'truncatedSVD'
            ytt = tt_tensor(Y,TT_tol,szy,rank_y);
        case 'cross'
            % This part is used for the dmrg_cross_sparsedata
            %             fun_idx = @(i) Y(szi);
            %             ytt = dmrg_cross_sparsedata(ndims_y,szy,fun_idx,TT_tol,...
            %                  'nswp',5,'vec',true,'kickrank',2);
            
            %             ytt = dmrg_cross_sparsedata(ndims_y,szy,fun_idx,TT_tol,...
            %                 'maxr',max(rank_y)+100,'nswp',5,'vec',true,'kickrank',2);
            
            % This part is used for the dmrg_cross
            csz = cumprod(szy);csz = csz(:);
            fsub2ind = @(i) (i(:,2:end)-1)*csz(1:end-1)+i(:,1);
            fun_idx = @(i) Y(fsub2ind(i));
            ytt = dmrg_cross(ndims_y,szy,fun_idx,TT_tol,...
                'maxr',max(rank_y)+150,'nswp',5,'vec',true,'kickrank',2);
    end
    truncate_y = true;
    if truncate_y
        %ytts = truncate(TT_to_TTeMPS(ytt),rnky);
        %ytt = TTeMPS_to_TT(ytts);
        ytt = round(ytt,TT_tol,rank_y);
    end
    % ytt = TTeMPS_to_TT(orthogonalize(TT_to_TTeMPS(ytt),dx+1));
else
    ytt = Y;
end

%% Initialization
xk = cell(R,1);
if iscell(param.init)
    xk = param.init;
    % correct size of initialization
    if no_mixtures > 1
        for k = 1:R
            xk{k} = reshape(xk{k},[size(xk{k}) 1]);
            % expand one more mode of lenght m for x_r
            xk{k} = ttm(xk{k},ndims(xk{k}),rand(1,no_mixtures));
        end
    end
    
elseif ischar(param.init)
    switch param.init
        case 'auto'
            xk = cell(R,1);
            
        case {'random' 'rand'}
            
            szx = szy;
            if no_mixtures > 1
                rank_x(:,ndims_x+1) = 1;
                rank_x(:,ndims_x+2) = 1;
            end
            
            xk = cell(R,1);
            for k = 1:R
                xk{k} = tt_rand(szx,numel(szx),rank_x(k,:));
            end
    end
end


%% Implementation 1
running_met_1 = true;
running_met_2 = false;
  
if running_met_1
    err = zeros(1,R);
    curr_err = [];
    ixR = find(cellfun(@(x) isempty(x),xk));
    ixR = [ixR(:)' setdiff(1:R,ixR)];
    for kiter = 1:param.maxiters
        
        if kiter >1
            err(kiter,:) = curr_err;
        end
        
        for k = ixR
            y_k = [];
            for l = setdiff(1:R,k)
                y_k = y_k + xk{l};
            end
            
            % compute the residue
            
            if isa(ytt,'tt_tensor')
                y_r = ytt - y_k;
            else
                if ~isempty(y_k)
                    y_r = ytt(:) - full(y_k);
                else
                    y_r = ytt(:) ;
                end
                y_r = tt_tensor(y_r,TT_tol,szy,rank_y);
            end
            
            % Fit a TT-tensor of rank rank_x to the residue y_r
            %x_r = TTeMPS_to_TT(truncate(TT_to_TTeMPS(y_r),rank_x(k,:)));
            % Truncation does not work well in some cases
            %x_r = round(y_r,TT_tol,rank_x(k,:));
            x_r = tensor_denoising(y_r,param.decomposition_method,param.noise_level,rank_x(k,:),xk{k});
            x_r = TTeMPS_to_TT(x_r);
            
            new_err = norm(y_r - x_r);
            if new_err >curr_err
                x_r = tt_als(y_r,x_r,10);
                new_err = norm(y_r - x_r);
            end 
            
            %         % Evaluate the new error
            %         new_err = norm(y_r - x_r);
            
            %         % The new error in general is smaller than the previous one.
            %         % However, due to compression or truncation of the TT-tensor ytt,
            %         % it may happen that the new error is not as expected.
            %         if ~isempty(curr_err) && (new_err > curr_err*(1+1e-2))
            %             % skip this update
            %             err(kiter,k) = curr_err;
            %         else
            curr_err = new_err;
            err(kiter,k) = curr_err;
            xk{k} = x_r;
            %         end
            fprintf('%d, %d -- %s\n',kiter,k,sprintf('%.3d ',err(kiter,:)))
        end
        
        if (kiter>1)
            deff = abs(diff(err(kiter-1:kiter,:)));
            if norm(deff)< param.tol
                break
            end
            
            % Update order
            [foe,ixR] = sort(deff,'descend');
        end
    end
end

%% Implementation 2

if running_met_2
    % Precomputing the error
    y_k = [];
    for l = 1:R
        y_k = y_k + xk{l};
    end
    if isa(ytt,'tt_tensor')
        y_r = ytt - y_k;
    else
        if ~isempty(y_k)
            y_r = ytt(:) - full(y_k);
        else
            y_r = ytt(:) ;
        end
        y_r = tt_tensor(y_r,TT_tol,szy,rank_y);
    end
    
    err = zeros(1,R);
    curr_err = [];
    ixR = find(cellfun(@(x) isempty(x),xk));
    ixR = [ixR(:)' setdiff(1:R,ixR)];
    for kiter = 1:param.maxiters
        
        if kiter >1
            err(kiter,:) = err(kiter-1,k);
        end
        
        for k = ixR
            
            y_r = y_r + xk{k};
            %         if isempty(xk{k})
            x_r = round(y_r,TT_tol,rank_x(k,:));
            
            
            %         else
            %             x_r = tt_als(y_r,xk{k},10);
            %         end
            %y_r = y_r - x_r;
            
            
            new_err = norm(y_r - x_r);
            if new_err > curr_err
                x_r = tt_als(y_r,x_r,10);
                y_r = y_r - x_r;
                new_err = norm(y_r);
            else
                y_r = y_r - x_r;
            end
            
            
            % Evaluate the new error
            
            % The new error in general is smaller than the previous one.
            % However, due to compression or truncation of the TT-tensor ytt,
            % it may happen that the new error is not as expected.
%             if ~isempty(curr_err) && (new_err > curr_err*(1+1e-2))
%                 % skip this update
%                 err(kiter,k) = curr_err;
%             else
                curr_err = new_err;
                err(kiter,k) = curr_err;
                xk{k} = x_r;
%             end
            fprintf('%d, %d -- %s\n',kiter,k,sprintf('%.3d ',err(kiter,:)))
            
        end
        
        y_r = round(y_r,1e-9);
        
        if (kiter>1)
            deff = abs(diff(err(kiter-1:kiter,:)));
            if norm(deff)< param.tol
                break
            end
            
            % Update order
            [foe,ixR] = sort(deff,'descend');
        end
    end
    
end

%%

if no_mixtures> 1
    
    % return the mixing matrix A
    Ah = cell2mat(cellfun(@(x) x{ndims_x+1},xk,'uni',0))';
    
    for k = 1:R
        cc = core2cell(xk{k});
        xk{k} = TTeMPS_to_TT(TTeMPS(cc(1:ndims_x)'));
    end
else
    Ah  = [];
end


end

%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','auto',@(x) iscell(x) || ismember(x(1:4),{'auto' 'rand'}));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('tt_fit_type','truncatesvd',@(x) ismember(x(1:4),{'none'}) || ismember(x(1:5),{'trunc' 'cross' }));

param.addOptional('decomposition_method','ttmps_adcu'); %tt_truncation ttmps_ascu  ttmps_ascu  ttmps_adcu ttmps_atcu
param.addOptional('noise_level',[]); 


% param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
%     ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));

param.parse(opts);
param = param.Results;
end
