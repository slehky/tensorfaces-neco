%% SINGLE GROUP, sequential shift

function [Yh,output,Yhp,Yshift] = ktdo_sgrp_seq_shift(Y,Patch_Opts,Yh,options,step,Noshifts)
% Yh is inital value, and can be an array of the same size of the data, or
% a Kruskal tensor of the unfolding tensor.


SzY = size(Y);
Yshift = zeros(prod(SzY), Noshifts);

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


patchsize = Patch_Opts.Size;
% Fix the patch sizes because of sequential shift
patchsize = cellfun(@(x) x(1:ndims(Y)),patchsize,'uni',0);
Patch_Opts.Size = patchsize;

for kshift = 1:Noshifts
    
    shift_subidx = ind2sub_full(2*step+1,kshift);
    shift_subidx = shift_subidx - step -1;
    fprintf('Shift %d, (%s)\n', kshift,sprintf('%d,',shift_subidx))
    
    % Shift data
    Ysh = circshift(Y,shift_subidx);
    Yh_sh = Yh;
    if ~isempty(Yh)  && isnumeric(Yh)
        Yh_sh = circshift(Yh,shift_subidx);
        % Don't need to shift Yh when it is a K-tensor.
    end
    
    % KTD to the shift data
    [Yh_sh,outputsh,Yhp_sh] = ktdo_sgrp(Ysh,Patch_Opts,Yh_sh,options);
    
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

Yh = options.shiftfilter(Yshift,2);
Yh = reshape(Yh,size(Y));
Yshift = reshape(Yshift,[SzY Noshifts]);
end

