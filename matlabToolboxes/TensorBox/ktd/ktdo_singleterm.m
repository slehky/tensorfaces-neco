function varargout = ktdo_singleterm(Y,Patch_Opts,opts)
% Kronecker tensor decomposition with single approximation term
%    Y ~~ sum_r  A_r1 \ox A_r2 \ox ... \ox A_rN
%
% where \ox denotes the Kronecker tensor product.
% Tensors A_rn are of the same size In(1) x In(2) x ... x In(N),
% for r = 1, ..., R
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
No_Groups = numel(Patch_Opts); % SHOULD BE ONE for single approximation term

%% Correct sizes of patches, sizes of the last patches will be corrected
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

%% Prepare linear transformation operators for groups of patches
Patch_Opts = linear_transform(Patch_Opts);

%% Select single or multi-term KTD
output = [];

if Noshifts == 0 % Standard single-approximation term KTD without shift
    normY = norm(Y(:));
    param.normX = normY;
    
    [Yh,output,Yhp] = ktdo_sgrp(Y,Patch_Opts,[],param);
    
    Yh = reshape(Yh,[Data_Sz No_Groups]);
    
    
else % Convolutive-like Kronecker tensor decomposition by shifting data in small interval
    
    % If data is expanded by shift, there is one more mode.
    % Correct sizes of patches. The first patch is extended one more
    % dimension
    % No_Groups is only 1
    for kp = 1:No_Groups
        Patch_Sizes{kp} = cellfun(@(x) [x 1],Patch_Sizes{kp},'uni',0);
        Patch_Sizes{kp}{1}(end) = Noshifts;
        Patch_Opts(kp).Size = Patch_Sizes{kp};
    end
     
    
    %% For single group
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
    end
end

%%
function shiftfilter = shiftfilter_op(shiftfilter)
switch shiftfilter
    case 'mean'
        shiftfilter = @(x,dim) mean(x,dim);
    case 'median'
        shiftfilter = @(x,dim) median(x,dim);
    case 'bestapprox'
        shiftfilter = @(x,dim) bestaprox(x);
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
param.addOptional('autothresh',false);
param.parse(opts);
param = param.Results;
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
