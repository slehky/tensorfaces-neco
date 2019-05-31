% Example for decomposition of a tensor of 6x6x6 and rank 12.
% 
% The tensor composes two tensors, each of rank-6.
% 
% The tensor is general diffuclt to fit by CPD because the tensor rank
% exceeds the tensor dimensions and the loading components are highly
% collinear.
%
% 1- The ALS first decomposed the tensor in 10 iterations to give initial
%    value.
% 2- The fLM algorithm then decomposes the tensor
%    a) using the tensor estimated by ALS as input
%    b) as above but with EPC to correct the norm of rank-1 tensor
%    components
%
% TENSORBOX 2017
%
%%
clear all
warning off
N = 3;   % tensor order
R = 6;   % tensor rank
szY = R*ones(1,N); % tensor size
noblks = 2;

% Generate a Kruskasl tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
c = [0.95 0.999;  0.95 0.999;  0.95 0.999;  0.95 0.999; 0.95 0.999;0.955 0.999];

A = cell(N,1);
for n = 1:N
    An = [];
    for k = 1:noblks
        An = [An  gen_matrix(szY(n),R,c(n,:))];
    end
    A{n} = An;
end
R = R*noblks;

% Add Gaussian noise
% lambda = rand(R,1);
lambda = ones(R,1);
Y = ktensor(lambda,A(:));  % a Kruskal tensor of rank-R

Y = normalize(Y);
rank1normbound = norm(Y.lambda);

Y = tensor(Y);
Rx = R;

%% STAGE 1: run ALS in a few iterations (10, 50  iterations)

opts = cp_fastals;
opts.init = {'dtld' 'nvec'  'nvec' 'rand'};
% opts.init = repmat({'rand'},1,10);
% opts.init = Pi;
opts.printitn = 1;
opts.maxiters = 10;
opts.linesearch = 0;
% opts.maxiters = 50;
opts.TraceRank1Norm = true;

opts.tol = 1e-9;
[P_als,out_als] = cp_fastals(tensor(Y),Rx,opts);

% result in the first stage
U = cellfun(@(x) bsxfun(@times,x,P_als.lambda'.^(1/3)),P_als.u,'uni',0);
P_als = ktensor(U);

%% STAGE 1b: Continue the CPD using the LM algorithm: CP-ALS -> fLM
opts = cp_fLMa();
opts.init = P_als;  % set P_als as the initial value 
opts.linesearch = 0;
opts.alsinit = 0;
opts.tol = 1e-15;
opts.maxiters = 1000;
opts.printitn = 1;
opts.TraceRank1Norm = true;

% fLM for CPD
[Pbx,outputx] = cp_fLMa(tensor(Y),Rx,opts);
Output = (1-real(outputx.Fit(:,2))).^2;
Rank1Norm = (sum(outputx.Rank1Norm.^2,2));
P_fLM0 = Pbx; % result obtained using fLM
 
%%  STAGE 2:   Run the Error Preserving Correction method to the initial tensor 

opts = cp_anc;
opts.maxiters = 2000;
opts.printitn = 1;
opts.linesearch = 0;
opts.tol = 1e-10;

opts.init = P_als;
delta = norm(Y(:))^2 + norm(P_als)^2 - 2*innerprod(Y,P_als);
delta = sqrt(delta);

[P_als_bals,output_als_bals] = cp_anc(Y,R,1.01*delta,opts);

%%  STAGE 2b:  Run fLM after the EPC 
%  CP-ALS ->  EPC -> fLM 

opts = cp_fLMa();
opts.init = P_als_bals;
opts.alsinit = 0;
opts.tol = 1e-15;
opts.maxiters = 3000;
opts.linesearch = 0;
opts.ineqnonlin = true;
opts.printitn = 1;
opts.TraceRank1Norm = true;

[Pbx,outputx] = cp_fLMa(tensor(Y),Rx,opts);
Output_normcorrection = (1-real(outputx.Fit(:,2))).^2;
Rank1Norm_normcorrection = (sum(outputx.Rank1Norm.^2,2)); 

%%  PLOT and compare results 
% generate figure 5 b
fig = figure(1);
clf
kistart = 1;
clear h;
h(1) = semilogy(sqrt([real(1-out_als.Fit(:,2)).^2 ; Output(kistart:end)])); % ALS - fLM
hold on
h(2) = semilogy(sqrt([real(1-out_als.Fit(:,2)).^2 ;Output_normcorrection(kistart:end)])); % ALS -> EPC -> fLM
h(3) = semilogy(real(1-out_als.Fit(:,2))); % ALS -> EPC -> fLM

set(gca,'YScale','log','XScale','log')

set(h,'linewidth',4)
set(h(2),'linestyle','--')
set(h(3),'linestyle','--')

ylabel('Relative Error')
xlabel('Iterations')

set(gca,'fontSize',16)

algs_name = {'ALS->fLM' 'ALS->EPC->fLM' 'ALS (init)'};
hlg = legend(h,algs_name,'location','best');
axis tight
grid on
set(gca,'XMinorGrid','off')
set(gca,'XTickLabelMode','auto')
 

set(hlg,'fontSize',16)

% 
% 
% figname = sprintf('res_mc_bcd_R%d_cpd_rel_vs_iter_fixed_20062018',R);
% saveas(fig,[figname '.fig'],'fig')
% print(fig,'-depsc',[figname '.eps'])
% 
% END OF THIS FILE here 
 