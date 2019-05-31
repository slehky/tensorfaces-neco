function  varargout = ktc_svt(Y,Ix,opts)
% Singular Value Thresholding for single group Kronecker tensor completion
%
% Input:
%   Y   :  data tensor with nan respresenting missing entries in Y.
%   Ix  :  row array indicates size of tensors X_r.
%   opts:  parameters of the decomposition.
%          Run the algorithm without any input to get the default parameter
%          of the algorithm:
%          opts = ktc_svt;
%
% Output
%  - For KTC without shift
%   [Yh,output] = ktc_svt(Y,Ix,opts);
%
%  where Yh is an approximate to Y. 
%  The "output" containts cost values, relative errors and ranks of X
%  during the estimation process.
%      output.cost, output.relerror and output.rank
%
%  - For KTC with shift
%   [Yh,output,Yshift] = ktc_svt(Y,Ix,opts);
%
%  where Yshift is an array of estimated images with different shifts.
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
%
% This file is a part of the TENSORBOX, 2014.

if ~exist('opts','var'),  opts = struct; end
param = parseinput(opts);

nargout_ = nargout;
if nargin == 0
    varargout{1} = param; return
end

SzY = size(Y);
Ia = bsxfun(@rdivide,SzY,Ix); % size of tensors A

step = param.step;
Noshifts = 0;
if ~isempty(step) && any(step >0)
    Noshifts = prod(2*step+1);
end

if Noshifts == 0
    
    % Rearrange Y to a matrix using the Kronecker tensor unfolding
    Y = kron_unfolding(Y,Ix);
    
    % SVT for the Kron_unfolding matrix
    [Yh,output] = svt_pg(Y, param.lambda, param.gamma, param.maxiters, param.init,param.tol);
    
    % Fold matrix back to tensor
    Yh = kron_folding(Yh,Ix,Ia);Yshift =  [];
    
else
    switch param.shift_type
        case 'sequential'
            if nargout_ >=2
                output = struct('cost',[],'relerror',[],'rank',[]);
                output = output(ones(Noshifts,1));
            end
            
            Yshift = zeros(prod(SzY), Noshifts);
            Yinit = param.init;
            Yinitsh = [];
            parfor kshift = 1:Noshifts
                
                [r,c] = ind2sub(2*step+1,kshift);
                r = r - step(1) -1; c = c - step(2) -1;
                fprintf('Shift %d, (%d,%d)\n', kshift,r,c)
                
                % Shift data
                Ysh = circshift(Y,[r c]);
                Yinitsh = [];
                if ~isempty(Yinit)
                    Yinitsh = circshift(Yinit,[r c]);
                end
                
                % Rearrange Y to a matrix using the Kronecker tensor unfolding
                Ysh = kron_unfolding(Ysh,Ix);
                
                % SVT to the shift data
                [Yh,outputsh] = svt_pg(Ysh, param.lambda, param.gamma, param.maxiters,Yinitsh,param.tol);
                
                if nargout_ >=2
                    output(kshift)= outputsh;
                end
                
                % Fold matrix back to tensor
                Yh = kron_folding(Yh,Ix,Ia);
                
                % Shift data back
                Yh = circshift(Yh,-[r c]);
                
                Yshift(:,kshift) = Yh(:);
            end
            
            Yh = median(Yshift,2);
            Yh = reshape(Yh,size(Y));
            Yshift = reshape(Yshift,[SzY Noshifts]);
            
        case 'augmented'
            
            % Construct tensors from data which shifts left-right in range
            % [-step(1):step(1)], and up-down in [-step(2):step(2)].
             
            Yshift = zeros(prod(SzY), Noshifts);
              
            for kshift = 1:Noshifts
                shift_ix = ind2sub_full(2*step+1,kshift);
                shift_ix = shift_ix - step - 1;
                Yshift(:,kshift) = reshape(circshift(Y,shift_ix),[],1);
            end
            Yshift = reshape(Yshift,[SzY Noshifts]);
            
            Ix(end+1) = 1;Ia(end+1) = Noshifts;
            % Rearrange Y to a matrix using the Kronecker tensor unfolding
            Yshift = kron_unfolding(Yshift,Ix);
            
            [Yh,output] = svt_pg(Yshift, param.lambda, param.gamma, param.maxiters,[],param.tol);
            
            
            % Fold matrix back to tensor
            Yh = kron_folding(Yh,Ix,Ia);
            
            %% Shift back approximate images
            Yh = reshape(Yh,prod(SzY),[]);
            for kshift = 1:size(Yh,2)
                shift_ix = ind2sub_full(2*step+1,kshift);
                shift_ix = shift_ix - step - 1;
                Yks = reshape(Yh(:,kshift),SzY);
                Yks = circshift(Yks,-shift_ix);
                Yh(:,kshift) = Yks(:);
            end 
            
            Yshift = Yh;
            Yh = median(Yshift,2);
            Yh = reshape(Yh,SzY);
            Yshift = reshape(Yshift,[SzY size(Yshift,2)]);
    end
end


nargout_m = nargout_;
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
        if (Noshifts>=1) && (nargout_m>=3)
            out{3} = Yshift;
        end
    end

end


function param = parseinput(opts)
param = inputParser;
param.addOptional('init',[]);
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('verbose',0);
param.addOptional('maxmse',50);
param.addOptional('step',[]);
param.addOptional('shift_type','sequential',@(x) ismember(x(1:4),{'sequ' 'augm'})); % sequential or aumgented processing

param.addOptional('normA',0);
param.addOptional('smoothA',0);

param.addOptional('lambda',.5);
param.addOptional('gamma',.5);


param.parse(opts);
param = param.Results;
end