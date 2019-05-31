function Xt = tt_kickrank(Y,Xt)
% Increase rank of the TT-tensor X
%% 
%
% TENSORBOX, 2018

rankX = Xt.rank; 
[rnk_s,mode_ix] = sort(rankX(2:end-1),'ascend');

%[rm,nm] = min(rankX(2:end-1));
for ki = 1:numel(rnk_s)
    nm = mode_ix(ki);
    rm = rnk_s(ki);
    sz1 = size(Xt.U{nm});sz2 = size(Xt.U{nm+1});
    if min(prod(sz1(1:2)),prod(sz2(2:end))) > rm
        break
    end
end

%% 
%  | Y - X|_F = |Y - Phi_<n x Un x U(n+1) x Phi>(n+1)|_F
%             = |Z - Un x U(n+1)|_F
%
% 
modes = [nm nm+1];
Z = ttxt(Xt,Y,modes,'both');
Z = reshape(Z,sz1(1)*sz1(2),[]);
 
% best (rm+1) rank approximation to the matrix Z
[u,s,v]=svd(Z,'econ');
% s=diag(s);
U1 = reshape(u(:,1:rm+1),sz1(1),sz1(2),[]);
U2 = reshape((v(:,1:rm+1)*s(1:rm+1,1:rm+1))',rm+1,sz2(2),[]);

%%
Xt.U{nm} = U1;
Xt.U{nm+1} = U2;
% rankX(nm) = rankX(nm)+1;
