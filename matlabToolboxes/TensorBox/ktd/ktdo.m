function varargout = ktdo(Y,Patch_Opts,opts)
% Kronecker tensor decomposition with orthogonal parts
%    Y = Y_1 + Y_2 + ... + Y_P
%each part Yp is composed by two sub-tensors Ap and Xp,
%    Y_p = A_p1 \ox  X_p1 + ... + A_pR \ox  X_pR
%
% where \ox denotes the Kronecker tensor product.
% Tensors X_pr in the same group indicated by index p,
% have the same size Ix(p1) x Ix(p2) x ... x Ix(pN), for r = 1, ..., Rp
% whereas tensors in different group, i.e. different p, have different
% sizes.
%
%
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

% Input:
%   Y   :  data tensor with nan respresenting missing entries in Y.
%   Ix  :  row array indicates size of tensors X_r.
%   R   :  number of tensors X_r.
%   opts:  optional parameters of the decomposition. The parameters are
%          similar to those of routines for single group and multigroup KTC
%          ktc_nng_sgrp and ktc_nng_mgrp.
%
%          Run the algorithm without any input to get the default parameter
%          of the algorithm:
%          opts = ktc_nng;
%
%      opts.normA  (0)  parameter for the regularized term || A_r ||_F^2
%      opts.smoothA (0)    parameter controls smoothness constraints on A_r
%      opts.maxmse (50 dB) algorithm will stop when MSE (dB) > 30 dB
%      opts.maxiters (200) maximum number of iterations.
%      opts.step (0)       when step is non zero, KTC is applied to the
%                          data shifted in an interval [r,s], |r|<= step.
%
%                          For example, step = [2 2], data will be shifted
%                          in an interval of [5,5].

%      opts.shift_type ('sequential','augmented') used when opts.step is
%                          non-zero
%        'sequential'      KTC sequentially factorizes data arrays
%                          shifted from the orignal one.
%                          For example, step = [2 2], there are 25
%                          data arrays generated from the observed data.
%                          Each shift data will be processed separately.
%
%        'augmented'       KTC will factorize the new augmented data which
%                          is generated from all shifts.
%      opts.multigroup_type   specify the processing method for multigroup
%                          KTC.
%         'sequential'     consider multigroup KTC as multiple KTC for
%                          singles groups.
%         'simultaneous'   process KTC for multigroup as it is.
%
% Output
%  - For KTC without shift
%   [Yh,Ah,Xh,rmse] = ktc_nng(Y,Ix,R,opts);
%
%  where Yh is an approximate to Y by Ah and Xh, rmse is relative MSEs.
%
%  - For KTC with shift
%   [Yh,Yshift] = ktc_nng(Y,Ix,R,opts);
%
%  where
%     Yh is an approximate to Y.
%     Yshift is an array of estimated images with different shifts
%
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
% Copyright Phan Anh Huy 2014-11
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by Phan Anh Huy.

if ~exist('opts','var'),  opts = struct; end
param = parseinput(opts);

if nargin == 0
    varargout{1} = param; return
end

%% Verify Patch structures
% Patch_struct = struct('Size',[],'Transform','none','No_comps',[],'Constraints',[],'Regularized_par',[]);
No_Groups = numel(Patch_Opts);
  
%% Correct sizes of the last patches
Data_Sz = size(Y);
[Patch_Opts,Patch_Sizes] = correct_patch_size(Data_Sz,Patch_Opts);

%% Parameters for the convolutive-like KTD
step = param.step;
Noshifts = 0;
if ~isempty(step) && any(step >0)
    Noshifts = prod(2*step+1);
    
    if Noshifts>1
        param.shiftfilter = shiftfilter_op(param.shiftfilter);
    end
    
    %%  Check if simultaneous shift will yield a big data, swith to sequential shift
    max_data_length = 2e9/8;
    if Noshifts*prod(Data_Sz) > max_data_length
        param.shift_type = 'sequential';
    end
end


%% Correct parameters for multiple approximation terms
if No_Groups > 1          % multiple group
    if numel(param.gamma) == 1
        param.gamma = param.gamma(ones(1,No_Groups));
    end
end

%% Prepare linear transform operators for groups of patches
Patch_Opts = linear_transform(Patch_Opts);
 
%% Select single or multi-term KTD
output = [];

if Noshifts == 0 % Standard single-approximation term KTD without shift
    normY = norm(Y(:));
    param.normX = normY;
    
    if No_Groups > 1          % multiple group, multi-approximation terms 
%         switch param.multigroups_solver
%             case 'alg'
%                 [Yh,output,Yhp] = ktdo_mgrps_alg(Y,Patch_Opts,[],[],param);
%             otherwise
                [Yh,output,Yhp] = ktdo_mgrps(Y,Patch_Opts,[],[],param);
%         end
        
    else  % size(Ix,1) == 1 : single group
        %param.nonnegativity = true;
        
        [Yh,output,Yhp] = ktdo_sgrp(Y,Patch_Opts,[],param);
    end
    Yh = reshape(Yh,[Data_Sz No_Groups]);
    %Yshift =  [];
    
else % Convolutive-like Kronecker tensor decomposition by shifting data in small interval
    
     
    % If data is expanded by shift, there is one more mode.
    % Correct sizes of patches. The first patch is extended one more
    % dimension
    
    for kp = 1:No_Groups
        Patch_Sizes{kp} = cellfun(@(x) [x 1],Patch_Sizes{kp},'uni',0);
        Patch_Sizes{kp}{1}(end) = Noshifts;
        Patch_Opts(kp).Size = Patch_Sizes{kp};
    end
    

    if No_Groups == 1
        % For single group
        switch param.shift_type
            case 'sequential' % sequential shift for single group
                normY = norm(Y(:));
                param.normX = normY;
                [Yh,output,Yhp] = ktdo_sgrp_seq_shift(Y,Patch_Opts,[],param,step,Noshifts);
                
            case {'simultaneous' 'augmented'} % simultaneous shift for single group
                param.normX = []; % norm of the shift data will be computed and assigned after shift
                
                [Yh,output,Yhp] = ktdo_sgrp_sim_shift(Y,Patch_Opts,[],param,step);
                
        end
        
        Yh = reshape(Yh,Data_Sz);
        
    else % FOR MULTIPLE GROUPS
        
        switch param.shift_type
            case 'sequential'
                normY = norm(Y(:));
                param.normX = normY;
                
                [Yh,output,Yhp] = ktdo_mgrps_seq_shift(Y,Patch_Opts,param,step,Noshifts);
                  
            case {'simultaneous' 'augmented'}
                param.normX = []; % norm of the shift data will be computed and assigned after shift
                [Yh,output,Yhp] = ktdo_mgrps_sim_shift(Y,Patch_Opts,param,step);
        end
        
        Yh = reshape(Yh,[Data_Sz No_Groups]);
    end
end

nargout_m = nargout;
out = set_output;
varargout(1:numel(out)) = out;
for k = numel(out)+1:nargout_m
    varargout{k} = [];
end

    function out = set_output
        if nargout_m>=1
            out{1} = Yh;
        end
        if nargout_m>=2
            out{2} = output;
        end
        if nargout_m>=3
            out{3} = Yhp;
        end
        if nargout_m>=4
            out{4} = Patch_Sizes;
        end
        %         if (Noshifts>=1) && (nargout_m>=3)
        %             out{3} = Yshift;
        %         end
    end


     
end



%%
function param = parseinput(opts)
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init',[]);
param.addOptional('maxiters',200);
param.addOptional('multigroup_updatecycles',200);
param.addOptional('tol',1e-6);
param.addOptional('verbose',0);
param.addOptional('maxmse',50);
param.addOptional('step',[]);
param.addOptional('shift_type','simultaneous',@(x) ismember(x(1:4),{'sequ' 'simu' 'augm'})); % sequential or aumgented processing
param.addOptional('multigroup_type','simultaneous',@(x) ismember(x,{'simultaneous' 'sequential'})); % sequential or aumgented processing
param.addOptional('multigroups_solver','alg',@(x) ismember(x,{'alg' 'hals'})); % sequential or aumgented processing

param.addOptional('Yref',[]);
param.addOptional('normX',[]);

param.addOptional('abs_tol',0);

param.addOptional('normA',0);
param.addOptional('smoothA',0);


% param.addOptional('lambda',.5);
param.addOptional('gamma',1);
param.addOptional('epsilon',inf); % noise power used as approximation error bound 

param.addOptional('linesearch',true); %% For tensor decompositions
 
param.addOptional('shiftfilter','bestapprox',@(x) ismember(x,{'mean' 'median' 'bestapprox'}));  
% param.addOptional('solver','lasso',@(x) ismember(x,{'bpdn' 'lasso' 'omp'}));
param.addOptional('autothresh',true);
param.parse(opts);
param = param.Results;
end


%%
function [Yh,output,Yh_p] = ktdo_mgrps(Y,Patch_Opts,Yh,Yh_p,options)
No_Patts = numel(Patch_Opts);
SzY = size(Y);
if isempty(Yh)
    Yh_p = cell(1,No_Patts);
    Yh = zeros([numel(Y),No_Patts]);
end

if ~isfield(options,'normX') || isempty(options.normX)
    normY = norm(Y(:));
    options.normX = normY;
end
gamma = options.gamma;
Fk_val = zeros(No_Patts,1);
Err = zeros(options.maxiters,1);

for kiter = 1:min(10,options.multigroup_updatecycles)
    
    fprintf('Iter %d\n',kiter);
    %%
    for kp = 1:No_Patts
        
        options.gamma = gamma(kp);
        % Check decomposition of approximation term
        % Low-rank Matrix factorization or low-rank CPD
        % Ell-1 minimization on sparse-induced transform domain
        
        Patch_curr = Patch_Opts(kp);
        
        % Orthogonal complement basis to the current term if orthogonal
        % constraint is set
        if Patch_curr.orthogonal_term == true
            Yref = Yh(:,[1:kp-1 kp+1:No_Patts]);
            options.Yref = Yref;
        else
            options.Yref = [];
        end 
        
        if numel(Patch_curr.Size) >= 2
            if Patch_curr.orthogonal_term == true
                % Fit data 
                [Yh_kp,output,Yh_p{kp}] = ktdo_sgrp(Y,Patch_curr,Yh_p{kp},options);
            else
                % Fit error between the data and all approximation terms
                % but the term to be updated
                E_kp = Y - reshape(sum(Yh(:,[1:kp-1 kp+1:end]),2),SzY);
                [Yh_kp,output,Yh_p{kp}] = ktdo_sgrp(E_kp,Patch_curr,Yh_p{kp},options);
            end
        else
            % Fit error between the data and all approximation terms
            % but the term to be updated
            E_kp = Y - reshape(sum(Yh(:,[1:kp-1 kp+1:end]),2),SzY);
            [Yh_kp,output,Yh_p{kp}] = ktdo_sgrp(E_kp,Patch_curr,Yh(:,kp),options);
        end
        Yh(:,kp) = Yh_kp(:);
        
        
        Fk_val(kp) =  output.Fk_val;
    end
    
    
    %% OBJECTIVE value
    %curr_obj = objective_eval(Y,Yh,Patch_Opts);
    Err(kiter) = output.Res_norm + sum(Fk_val);
    
    if (kiter > 1) && (abs(Err(kiter)-Err(kiter-1))< 1e-5*abs(Err(kiter-1)))
        break
    end
end
Err = Err(1:kiter);
output = struct('Res_norm',output.Res_norm,'Fk_val',Fk_val,'Iter',numel(Err),'Objectiv',Err);
end


%%

function objective = objective_eval(Y,Yh,Patch_Opts)
E = Y(:) - sum(Yh,2); % residue
objective = norm(E(:))^2;
SzY = size(Y);
for k = 1:size(Yh,2) % No_Patts
    
    if numel(Patch_Opts(k).Size)==2 && ...
            (strcmp(Patch_Opts(k).Constraints,'lowrank')||strcmp(Patch_Opts(k).Constraints,'sparse'))
        
        Yhk = Yh(:,k);
        Yhk = reshape(Yhk,SzY);
        Yhk = kron_unfoldingN(Yhk,Patch_Opts(k).Size);
        
        switch Patch_Opts(k).Constraints
            case 'lowrank'
                
                objective = objective + Patch_Opts(k).Regularized_par * norm_nuclear(Yhk);
            case 'sparse'
                
                WYhk = Patch_Opts(k).TF(Yhk);
                objective = objective + Patch_Opts(k).Regularized_par * norm(WYhk(:),1); % need to fix - ell-1 norm on the transform domain
        end
    end
end
end


%% Multiple GROUPs, sequential shift

function [Yh,output,Yhp,Yshift] = ktdo_mgrps_seq_shift(Y,Patch_Opts,options,step,Noshifts)
SzY = size(Y);
No_Groups = numel(Patch_Opts);
isreturn_output = false;
if nargout >=2
    output = struct('Res_norm',[],'Fk_val',[],'Iter',[],'Objectiv',[]);
    isreturn_output = true;
end
isreturn_grp_comps = false;
if nargout >=3
    Yhp = cell(Noshifts,1);
    isreturn_grp_comps = true;
end

ndimY = ndims(Y);
for kp = 1:numel(Patch_Opts)
    % Fix the patch sizes because of sequential shift
    Patch_Opts(kp).Size = cellfun(@(x) x(1:ndimY),Patch_Opts(kp).Size,'uni',0);
end

Yshift = zeros(prod(SzY)*No_Groups, Noshifts);

for kshift = 1:Noshifts    
    shift_subidx = ind2sub_full(2*step+1,kshift);
    shift_subidx = shift_subidx - step -1;
    fprintf('Shift %d, (%s)\n', kshift,sprintf('%d,',shift_subidx))
    
    % Shift data
    Ysh = circshift(Y,shift_subidx);
    
    % KTD to the shift data
%     switch options.multigroups_solver
%         case 'alg'
%             [Yh_sh,outputsh,Yhp_sh] = ktdo_mgrps_alg(Ysh,Patch_Opts,[],[],options);
%         otherwise
            [Yh_sh,outputsh,Yhp_sh] = ktdo_mgrps(Ysh,Patch_Opts,[],[],options);
%     end
    
    
    
    Yh_sh = reshape(Yh_sh,[SzY No_Groups]);
    % Shift data back
    Yh_sh = circshift(Yh_sh,-shift_subidx);
    
    Yshift(:,kshift) = Yh_sh(:);
    
    
    if isreturn_output
        output(kshift) = outputsh;
    end
    if isreturn_grp_comps
        Yhp{kshift} = Yhp_sh;
    end
end

% Median or Mean over shift
Yh = options.shiftfilter(Yshift,2);
Yh = reshape(Yh,[SzY No_Groups]);

if nargout >= 4
    Yshift = reshape(Yshift,[SzY No_Groups Noshifts]);
    Yshift = permute(Yshift,[1 3 2]);
end
end

%% Multiple GROUPs, simultaneous shift
function [Yh,output,Yhp,Yshift] = ktdo_mgrps_sim_shift(Y,Patch_Opts,options,step)
SzY = size(Y);
No_Groups = numel(Patch_Opts);

% SHIFT DATA
[Yshift,Noshifts] = kron_shift(Y,step);
% switch options.multigroups_solver
%         case 'alg'
%             [Yshift,output,Yhp] = ktdo_mgrps_alg(Yshift,Patch_Opts,[],[],options);
%         otherwise
            [Yshift,output,Yhp] = ktdo_mgrps(Yshift,Patch_Opts,[],[],options);
% end
    

Yshift = reshape(Yshift,[prod(SzY)  Noshifts No_Groups]);
Yshift = permute(Yshift,[1 3 2]);
Yshift = kron_ishift(Yshift,[SzY  No_Groups],step);
Yshift = reshape(Yshift,[],Noshifts);
Yh = options.shiftfilter(Yshift,2);
Yh = reshape(Yh,[SzY No_Groups]);

if nargout >= 4
    Yshift = reshape(Yshift,[SzY,No_Groups Noshifts]);
end
end