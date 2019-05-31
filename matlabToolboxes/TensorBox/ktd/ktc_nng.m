function varargout = ktc_nng(Y,Ix,R,opts)
% Single group Kronecker tensor decomposition with nonnegativity constraint.
%
% The input tensor Y is approximated as
%
%     Y = A_1 \ox  X_1 + ... + A_R \ox  X_R
%
% where \ox denotes the Kronecker product between two tensors A_r and X_r.
% All tensors X_r have the same size Ix(1) x Ix(2) x ... x Ix(N) (single
% group).
%
% For multiple group decomposition, see ktc_nng.m.
%
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
Nogroups = size(Ix,1);
SzY = size(Y);

step = param.step;
shiftdata = false;
if ischar(step) && strcmp(step,'auto')
    shiftdata = true;
end

if ~isempty(step) && (isnumeric(step) && any(step >0))
    shiftdata = true;
end


if shiftdata == false
    Noshifts = 0;
    
    if Nogroups > 1          % multiple group
        switch param.multigroup_type
            case 'simultaneous'
                [Ah,Xh,Yh,Yhp,A,X,rmse] = ktc_nng_mgrp(Y,Ix,R,param);
            case 'sequential'
                Yha = zeros([prod(SzY) Nogroups]);
                Ah = cell(Nogroups,1);Xh = cell(Nogroups,1);
                for kg = 1:Nogroups
                    [Ahg,Xhg,Yhg,rmse] = ktc_nng_sgrp(Y,Ix(kg,:),R(kg,:),param);
                    Yha(:,kg) = Yhg(:);
                    Ah{kg} = Ahg;Xh{kg} = Xhg;
                end
                Yh = mean(Yha,2);
                Yh = reshape(Yh,SzY);
        end
    else  % size(Ix,1) == 1 : single group
        [Ah,Xh,Yh,rmse] = ktc_nng_sgrp(Y,Ix,R,param);
    end
    
else % KTC with shift
    Noshifts = prod(2*step+1);
    switch param.shift_type
        case 'sequential'
            
            switch param.multigroup_type
                case 'simultaneous'
                    
                    Yshift = zeros(prod(SzY), Noshifts);
                case 'sequential'
                    Yshift = zeros(prod(SzY), Nogroups*Noshifts);
            end
            
            cnt = 1;
            for kshift = 1:Noshifts
                
                shift_ix = ind2sub_full(2*step+1,kshift);
                shift_ix = shift_ix - step - 1;
                fprintf('Shift %d, (%s)\n', kshift,sprintf('%d_',shift_ix))
                
                % Shift data
                Ysh = circshift(Y,shift_ix);
                
                if Nogroups > 1         % multiple group
                    switch param.multigroup_type
                        case 'simultaneous'
                            [Ah,Xh,Yh,Yhp,A,X,rmse] = ktc_nng_mgrp(Ysh,Ix,R,param);
                        case 'sequential'
                            Yh = zeros([prod(SzY) Nogroups]);
                            %Ah = cell(Nogroups,1);Xh = cell(Nogroups,1);
                            for kg = 1:Nogroups
                                [Ahg,Xhg,Yhg,rmse] = ktc_nng_sgrp(Ysh,Ix(kg,:),R(kg,:),param);
                                Yh(:,kg) = Yhg(:);
                                %Ah{kg} = Ahg;Xh{kg} = Xhg;
                            end
                            Yh = reshape(Yh,[SzY size(Yh,2)]);
                    end
                else
                    [Ah,Xh,Yh,rmse] = ktc_nng_sgrp(Ysh,Ix,R,param);
                end
                
                % Shift data back
                Yh = circshift(Yh,-shift_ix);
                Yh = reshape(Yh,prod(SzY),[]);
                Yshift(:,cnt:cnt+size(Yh,2)-1) = Yh;
                cnt = cnt +size(Yh,2);
            end
            Yh = median(Yshift,2);
            Yh = reshape(Yh,SzY);
            Yshift = reshape(Yshift,[SzY size(Yshift,2)]);
            
        case 'augmented'
            
            % Construct tensors from data which shifts left-right in range
            % [-step(1):step(1)], and up-down in [-step(2):step(2)].
%             switch param.multigroup_type
%                 case 'simultaneous'
            Yshift = zeros(prod(SzY), Noshifts);
%                 case 'sequential'
%                     Yshift = zeros(prod(SzY), Noshifts);
%             end
            
            
            for kshift = 1:Noshifts
                shift_ix = ind2sub_full(2*step+1,kshift);
                shift_ix = shift_ix - step - 1;
                Yshift(:,kshift) = reshape(circshift(Y,shift_ix),[],1);
            end
            Yshift = reshape(Yshift,[SzY Noshifts]);
            Ix(:,end+1) = 1;
            
            % KTD
            if Nogroups > 1         % multiple group
                switch param.multigroup_type
                    case 'simultaneous'
                        [Ah,Xh,Yh,Yhp,A,X,rmse] = ktc_nng_mgrp(Yshift,Ix,R,param);
                        
                        %% Shift back approximate images
                        Yh = reshape(Yh,prod(SzY),[]);
                        for kshift = 1:size(Yh,2)
                            shift_ix = ind2sub_full(2*step+1,kshift);
                            shift_ix = shift_ix - step - 1;
                            Yks = reshape(Yh(:,kshift),SzY);
                            Yks = circshift(Yks,-shift_ix);
                            Yh(:,kshift) = Yks(:);
                        end
                        
                    case 'sequential'
                        Yh = zeros([numel(Yshift) Nogroups]);
                        %Ah = cell(Nogroups,1);Xh = cell(Nogroups,1);
                        for kg = 1:Nogroups
                            [Ahg,Xhg,Yhg,rmse] = ktc_nng_sgrp(Yshift,Ix(kg,:),R(kg,:),param);
                            
                            
                            %% Shift back approximate images
                            Yhg = reshape(Yhg,prod(SzY),[]);
                            for kshift = 1:size(Yhg,2)
                                shift_ix = ind2sub_full(2*step+1,kshift);
                                shift_ix = shift_ix - step - 1;
                                Yks = reshape(Yhg(:,kshift),SzY);
                                Yks = circshift(Yks,-shift_ix);
                                Yhg(:,kshift) = Yks(:);
                            end
                            Yh(:,kg) = Yhg(:);
                            %Ah{kg} = Ahg;Xh{kg} = Xhg;
                        end
                        Yh = reshape(Yh,prod(SzY),[]);
                        %                         Yh = reshape(Yh,[SzY size(Yh,2)]);
                end
                
            else
                [Ah,Xh,Yh,rmse] = ktc_nng_sgrp(Yshift,Ix,R,opts);
                
                %% Shift back approximate images
                Yh = reshape(Yh,prod(SzY),[]);
                for kshift = 1:size(Yh,2)
                    shift_ix = ind2sub_full(2*step+1,kshift);
                    shift_ix = shift_ix - step - 1;
                    Yks = reshape(Yh(:,kshift),SzY);
                    Yks = circshift(Yks,-shift_ix);
                    Yh(:,kshift) = Yks(:);
                end
                
            end
            
            Yshift = Yh;
            Yh = median(Yshift,2);
            Yh = reshape(Yh,SzY);
            Yshift = reshape(Yshift,[SzY size(Yshift,2)]);
    end
    
end
nargout_m = nargout;
out = set_output;
varargout(1:numel(out)) = out;
for k = numel(out)+1:nargout_m
    varargout{k} = [];
end

    function out = set_output
        
        if Noshifts == 0
            if nargout_m>=1
                out{1} = Yh;
            end
            if nargout_m>=2
                out{2} = Ah;
            end
            if nargout_m>=3
                out{3} = Xh;
            end
            if nargout_m>=4
                out{4} = rmse;
            end
            
            if Nogroups >=2          % multiple group
                if nargout_m>=5
                    out{5} = A;
                end
                if nargout_m>=6
                    out{6} = X;
                end
                if nargout_m>=7
                    out{7} = Yhp;
                end
            end
            
        else %(Noshifts>=1)
            
            if nargout_m>=1
                out{1} = Yh;
            end
            if (nargout_m>=2)
                out{2} = Yshift;
            end
        end
    end
end


function param = parseinput(opts)
param = inputParser;
param.addOptional('init','nvec',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('verbose',0);
param.addOptional('maxmse',50);
param.addOptional('step','auto',@(x) (isnumeric(x)||ismember(x(1:4),{'auto' 'none'})))
param.addOptional('shift_type','sequential',@(x) ismember(x(1:4),{'sequ' 'augm'})); % sequential or aumgented processing
param.addOptional('multigroup_type','simultaneous',@(x) ismember(x,{'simultaneous' 'sequential'})); % sequential or aumgented processing

param.addOptional('normA',0);
param.addOptional('smoothA',0);

param.parse(opts);
param = param.Results;
end