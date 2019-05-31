function [U,patch_loc] = gen_surround_patches(Y,A,d_neibr,step_d,patch1st,dct_on,nonlinear_map,d_center)
% d_neibr: block width of neighbour patches
% d_center: block width of observed patches
% A : indicator
% Phan Anh Huy, 2017

if nargin < 8
    d_center = 1;
end

SzY = size(Y);
if ~isempty(A)
    K = sum(sum((A(d_neibr+d_center:SzY(1)-d_neibr-d_center+1,d_neibr+d_center:SzY(1)-d_neibr-d_center+1)))); % number of active pixels
else
    K = prod(SzY-2*(d_neibr+d_center-1));
end
patch_loc = zeros(K,2);
Uki = zeros(d_neibr^2,1,K); % d_neibr^2 x number_patches x K

kcnt = 0;
for ic = d_neibr+d_center:SzY(2)-d_neibr-d_center+1
    for ir = d_neibr+d_center:SzY(1)-d_neibr-d_center+1
        if (~isempty(A) &&  A(ir,ic)~=0) || isempty(A)
            kcnt = kcnt +1;
            [Urc,patchlocs] = gen_patches(Y,ir,ic,d_neibr,step_d,patch1st,d_center);
            Uki(:,1:size(Urc,2),kcnt) = Urc;
            
            patch_loc(kcnt,:) = [ir ic];
        end
    end
end

N = size(Uki,2);  % order of the weight tensor
K = size(Uki,3);  % number of feature vectors
U = mat2cell(reshape(permute(Uki,[1 3 2]),size(Uki,1),[]),size(Uki,1),K*ones(1,N));
U = U'; clear Uki;

% DCT
if dct_on
    ldic = dctmtx(d_neibr);
    rdic = dctmtx(d_neibr);
    Dict = kron(rdic,ldic);
    
    U = cellfun(@(x) Dict'*x,U,'uni',0);
end
% nonlinear
if nonlinear_map
    U = cellfun(@(x) sigmf(x,[1,0]),U,'uni',0);
end
end