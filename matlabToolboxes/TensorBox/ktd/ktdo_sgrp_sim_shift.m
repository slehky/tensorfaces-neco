%% SINGLE GROUP KTD, simultaneous shift

function [Yh,output,Yhp,Yshift] = ktdo_sgrp_sim_shift(Y,Patch_Opts,Yh,options,step)
SzY = size(Y);

% Shift data
[Yshift,Noshifts] = kron_shift(Y,step);
% Shift the reference or complement accordingly
if ~isempty(options.Yref)
    options.Yref = kron_shift(reshape(options.Yref,size(Y)),step);
    options.Yref = options.Yref(:);
end

[Yshift,output,Yhp] = ktdo_sgrp(Yshift,Patch_Opts,Yh,options);

Yshift = reshape(Yshift,prod(SzY),Noshifts);
Yshift = kron_ishift(Yshift,SzY,step);
Yh = options.shiftfilter(Yshift,2);
Yh = reshape(Yh,SzY);
Yshift = reshape(Yshift,[SzY Noshifts]);
end
