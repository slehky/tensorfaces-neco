% Approximation of the Hilber tensor using the TT-to-CPD method
% 
% TENSORBOX
% Phan Anh-Huy

clear all
warning off
N = 4;   % tensor order
I = 20*ones(1,N); % tensor size

ix = 1:prod(I);
is = ind2sub_full(I,ix);

% generate Hilbert tensor
Y = reshape(1./(sum(is,2)-N+1),I);
clear ix is; 

%% CPD - Decomposition with rank given 

Err_ = [];
R1norm_ = [];
RTime = [];
NoIters = [];
maxiters = 5000;
tol = 1e-10;

%%
for R = 1:8 % rank of the approximated tensor
    
    opts = cp_init;
    opts.init = {'nvecs' 'rand' 'rand'};
    
    Pi = cp_init(tensor(Y),R,opts);
    
    %% CPD
    opts = cp_fastals;
    opts.tol = tol;
    opts.maxiters = maxiters;
    opts.printitn = 1;
    opts.init = Pi;
    tic;
    [Pcp_als,output_cp_als] = cp_fastals(tensor(Y),R,opts);
    t = toc;
    
    Pcp_als = arrange(normalize(Pcp_als));
    R1norm_(1,R) = norm(Pcp_als.lambda);
    Err_(1,R) = norm(Y-full(Pcp_als)); %
    RTime(1,R) = t;
    NoIters(1,R) = size(output_cp_als.Fit,1);
    
    
    %% Algorithm :  LM-TT_conversion algorithm
    opts = cp_ttconv;
    opts.tol = tol;
    opts.maxiters = maxiters;
    opts.printitn = 1;
    tic;
    [P_tt2cp_lm,output_tt2cp_lm] = cpc_ttconv_fLM(Y,R,opts);
    %     7.0362e-05
    t = toc;
    
    Err_(2,R) = norm(Y-full(P_tt2cp_lm)); %  7.4811e-04
    P_tt2cp_lm = arrange(normalize(P_tt2cp_lm));
    R1norm_(2,R) = norm(P_tt2cp_lm.lambda);% 9.1910
    RTime(2,R) = t;
    NoIters(2,R) = size(output_tt2cp_lm.Fit,1);
end

%% Compare approximation errors 
close all
fig = figure(1);
clf

h = semilogy(nanmin(Err_,[],3)');
set(h,'Linewidth',4)
set(h(2),'LineStyle',':','Linewidth',4)


algs = {'ALS' 'TT2CP'}
legend(h,algs)
set(gca,'fontsize',15)

xlabel('Estimated Rank')
ylabel('Relative Error')
xlim(max(1,xlim))
% 
% fname = sprintf('fig_hilbert_I%dN%d_err',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')


% 
% %%Number of Iterations
% fig = figure(2);
% clf
% 
% h = semilogy(NoIters');
% set(h,'Linewidth',4)
% set(h(2),'LineStyle',':','Linewidth',4)
% set(h(3),'LineStyle','-.','Linewidth',6)
% 
% xlim(max(1,xlim))
% 
% algs = {'ALS' 'TTCP-ALS' 'TTCP-LM'}
% legend(h,algs,'location','northwest')
% set(gca,'fontsize',15)
% 
% xlabel('Estimated Rank')
% ylabel('No. Iterations')

% 
% fname = sprintf('fig_hilbert_I%dN%d_noiters',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')



% %% R1norm_
% close all
% fig = figure(1);
% clf
% 
% h = semilogy(R1norm_');
% set(h,'Linewidth',4)
% set(h(4),'LineStyle','--')
% set(h(2),'LineStyle',':','Linewidth',4)
% set(h(5),'LineStyle','-.','Linewidth',6)
% 
% 
% algs = {'ALS' 'fLM' 'TTCP-ALS' 'TTCP-LM' 'NLS'}
% legend(h,algs,'location','northwest')
% % delete(h(3))
% if N== 4
%     delete(h([3]))
% elseif N>4
%     delete(h([2 4]))
% end
% 
% set(gca,'fontsize',15)
% xlim(max(1,xlim))
% 
% xlabel('Rank-1 Norm')
% ylabel('Relative Error')
% 
% 
% fname = sprintf('fig_hilbert_I%dN%d_r1norm',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')

% %%  _rtime
% 
% close all
% fig = figure(1);
% clf
% 
% h = semilogy(RTime');
% set(h,'Linewidth',4)
% set(h(4),'LineStyle','--')
% set(h(2),'LineStyle',':','Linewidth',4)
% set(h(5),'LineStyle','-.','Linewidth',6)
% 
% xlim(max(1,xlim))
% 
% algs = {'ALS' 'fLM' 'TTCP-ALS' 'TTCP-LM' 'NLS'}
% legend(h,algs,'location','northwest')
% % delete(h(3))
% if N== 4
%     delete(h([3]))
% elseif N>4
%     delete(h([2 4]))
% end
% set(gca,'fontsize',15)
% 
% xlabel('Rank-1 Norm')
% ylabel('Running Time (s)')

% 
% fname = sprintf('fig_hilbert_I%dN%d_rtime',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')
% % 