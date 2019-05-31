%%  Test simple TT-tensor
Wt = sptenmat(W_resc, 1:K,K+1:2*K,[Sz1 Sz2]);
Wt = sptensor(Wt);
Wt = permute(Wt,dimp);
Wt = reshape(Wt,[1 prod(Sz1.*Sz2)]);
Wt = spmatrix(Wt);

W_tts = tt_stensor(Wt(:),tol,[Sz1(:) ; Sz2(:)],[1; 10*ones(2*K-1,1);1]);
W_tts = tt_tensor(W_tts);

norm(Wt(:) - full(W_tts))

%% Approximate P by a TT-matrix
fprintf('Approximate the affinity matrix by a TT-matrix\n')

tol = 1e-5;
% Sz1 = 4*ones(1,K);

% Sz1 = [16 4 4 4 16];
% Sz1 = [8 4 4 8];
Sz1 = [4 4 4 4];max_rank = 20;

Sz2 = Sz1;

K = numel(Sz1);

Wt = sptenmat(W_resc, K:-1:1,K+1:2*K,[Sz1 Sz2]);
Wt = sptensor(Wt);
Wt = reshape(Wt,[prod(Sz1.*Sz2) 1]);
Wt = spmatrix(Wt);


%% symmetric TT-tensor
W_tts = tt_stensor_sym(Wt(:),tol,[Sz1(:) ; Sz2(:)],[1; 10*ones(2*K-1,1);1]);
W_tts = tt_tensor(W_tts);

Wff = full(W_tts); 
Wff = reshape(Wff,size(W_tts));
Wff = ipermute(Wff,[K:-1:1 K+1:2*K]);
Wff  = reshape(Wff,size(W_resc));
norm(Wff - Wff','fro')

norm(Wt(:) - full(W_tts))

%%

% TT approximation
fun_idx = @(i) Wt(i);
 
W_tt = dmrg_cross_sparsedata_sym(K*2,[Sz1(end:-1:1) Sz2],fun_idx,tol,'maxr',max_rank,'nswp',5,'vec',true,'kickrank',0);

%%
Wff = full(W_tt); 
Wff = reshape(Wff,size(W_tt));
Wff = ipermute(Wff,[4 3 2 1 5:8]);
Wff  = reshape(Wff,size(W_resc));
norm(Wff - Wff','fro')