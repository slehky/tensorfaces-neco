function [msae,msae2,sae,sae2,src_reorder] = SAE(U,Uh,pospart)
% Square Angular Error
% sae: square angular error between U and Uh  
% msae: mean over all components
% 
% [1] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, "Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition",
% http://arxiv.org/abs/1209.3215, 2012. 
%
% [2] P. Tichavsky and Z. Koldovsky, Stability of CANDECOMP-PARAFAC
% tensor decomposition, in ICASSP, 2011, pp. 4164?4167. 
%
% [3] P. Tichavsky and Z. Koldovsky, Weight adjusted tensor method for
% blind separation of underdetermined mixtures of nonstationary sources,
% IEEE Transactions on Signal Processing, 59 (2011), pp. 1037?1047.
%
% [4] Z. Koldovsky, P. Tichavsky, and A.-H. Phan, Stability analysis and fast
% damped Gauss-Newton algorithm for INDSCAL tensor decomposition, in
% Statistical Signal Processing Workshop (SSP), IEEE, 2011, pp. 581-584. 
%
% TENSOR BOX, v1. 2012
% Phan Anh Huy, 2011

if nargin < 3
    pospart = 0;
end


N = numel(U);
R = size(U{1},2);
sae = nan(N,size(Uh{1},2));
sae2 = nan(N,R);
for n = 1: N
%     U{n} = bsxfun(@minus,U{n},mean(U{n}));
%     Uh{n} = bsxfun(@minus,Uh{n},mean(Uh{n}));
    
    C = U{n}'*Uh{n};
    C = C./(sqrt(sum(abs(U{n}).^2))'*sqrt(sum(abs(Uh{n}).^2)));
    % bug when a component is zero.
    nanC = isnan(C);
    C = acos(min(1,abs(C)));
    C(nanC) = nan;
    
    [sae(n,1:size(C,2)),rord1] = nanmin(C,[],1);
    [sae2(n,1:size(C,1)),rord2] = nanmin(C,[],2);
    
    if pospart == 1
        W = U{n} > 1e-4 *mean(U{n}(:));
        Uh{n} = Uh{n}(:,rord2);
        Uh{n} = Uh{n} .* W;
        
        C = U{n}'*Uh{n};
        C = C./(sqrt(sum(abs(U{n}).^2))'*sqrt(sum(abs(Uh{n}).^2)));
        C = acos(min(1,abs(C)));
        
        [sae(n,1:size(C,2)),rord1] = min(C,[],1);
        [sae2(n,1:size(C,1)),rord2] = min(C,[],2);
        
    end
end
sae = sae.^2;
sae2 = sae2.^2;
msae = nanmean(sae(:));

src_reorder = rord2;
msae2 = nanmean(sae2(:));