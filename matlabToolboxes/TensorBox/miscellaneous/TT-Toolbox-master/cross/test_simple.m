% TT approximation
W = reshape(randn(32,32),[],1);
% fun_idx = @(i) W(i);
 
% W_tt = dmrg_cross(10,2*ones(1,10),fun_idx,1e-4,'maxr',5,'nswp',5,'vec',true,'kickrank',0);


%% This part is used for the dmrg_cross
szy = 2*ones(1,10);
ndims_y = numel(szy)
csz = cumprod(szy);csz = csz(:);
fsub2ind = @(i) (i(:,2:end)-1)*csz(1:end-1)+i(:,1);
fun_idx = @(i) W(fsub2ind(i));
ytt = dmrg_cross(ndims_y,szy,fun_idx,1e-7,...
    'maxr',10,'nswp',5,'vec',true,'kickrank',2);