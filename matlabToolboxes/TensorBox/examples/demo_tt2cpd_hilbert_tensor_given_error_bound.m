% Approximation of the Hilber tensor using the TT-to-CPD method
%   i.e.   |Y - Yx|_F <= error_bound
%
% TENSORBOX
% Phan Anh-Huy

clear all
warning off
N = 4;   % tensor order
I = 10*ones(1,N); % tensor size

ix = 1:prod(I);
is = ind2sub_full(I,ix);

% generate Hilbert tensor
Y = reshape(1./(sum(is,2)-N+1),I);
clear ix is; 
normY = norm(Y(:));

%% CPD - Decomposition with a given error bound
% |Y - Yx|_F <= accuracy * |Y|_F

accuracy = 1e-3; % relative error bound

opts_2 = ttmps_a2cu;
error_bound = accuracy * normY; % |Y - Yhat|_F
opts_2.noise_level = .01*error_bound^2;  % the approximated TT-tensor has a lower apporoximation error bound
opts_2.compression = false;

% TT-decomposition using the ASCU algorithm
[Xt,output] = ttmps_ascu(double(Y),[],opts_2);

err_tt = norm(tensor(Y-full(Xt)))^2;

Xt = TTeMPS_to_TT(Xt);

%% Main process: CPD with bound 
% 
opts = cp_nc_itp;
opts.ineqnonlin= true;
opts.bound_ell2_constraint = false;
opts.printitn = 1;
opts.tol = 1e-8;
opts.maxiters = 2000;
opts.Algorithm = 'sqp';%'interior-point';
opts.history = false;

%%
for krun = 1:5
    for Rcp = 1:10
        %% LM algorithm for TT-2-CPD
        [P_tt2cp,output_tt2cp] = cpc_ttconv_fLM(Xt,Rcp,opts);
              
        % Alternating algorithm (ANC or ACEP) for EPC
        opts.init = P_tt2cp;
        [Pb_ipt,output_b_ipt] = cp_anc(tensor(Y),Rcp,error_bound,opts);
        opts.init = Pb_ipt;
        
        % SQP algorithm for EPC
        tic;
        [Pb_ipt,output_b_ipt] = cp_nc_sqp(tensor(Y),Rcp,error_bound,opts);
        t = toc;       
        % if the estimated rank R>20, using ANC or ITP: cp_nc_itp      
        
        % Assess performance 
        Pb_ipt = arrange(normalize(Pb_ipt));
        Err_(1,Rcp,krun) = norm(Y-full(Pb_ipt))/normY; %   0.0353
        R1norm_(1,Rcp,krun) = norm(Pb_ipt.lambda);%   3.8828
        RTime(1,Rcp,krun) = t;
        NoIters(1,Rcp,krun) = numel(output_b_ipt.fval);
    end
end 

%% EPC-SQP using SVD-based initialization method
opts = cp_nc_sqp;
opts.init = 'nvec';
opts.ineqnonlin = true;
opts.printitn = 1;

opts.maxiters = 2000;
opts.history = true;
opts.tol = 1e-10;

for krun = 1:5
    for Rcp = 1:10
        % opts.ineqnonlin = true;
        try
            [P_2,output2] = cp_nc_sqp(tensor(Y),Rcp,error_bound,opts);
            
            P_2 = arrange(normalize(P_2));
            Err_(2,Rcp,krun) = norm(Y-full(P_2))/normY;
            R1norm_(2,Rcp,krun) = norm(P_2.lambda);
            RTime(2,Rcp,krun) = t;
            NoIters(2,Rcp,krun) = numel(output2.fval);
        catch 
            continue
        end
    end
end

%% EPC using CP-ALS to provide initial point
opts = cp_nc_sqp;
opts.init = 'nvec';
opts.ineqnonlin = true;
opts.printitn = 1;

opts.maxiters = 2000;
opts.history = true;
opts.tol = 1e-8;

cp_opts = cp_fastals;
cp_opts.tol = 1e-10;
cp_opts.maxiters = 2000;
cp_opts.init = 'nvec';

for krun = 1:5
    for Rcp = 1:10
        tic;
        [Pcp_als,output_cp_als] = cp_fastals(tensor(Y),Rcp,cp_opts);
        t = toc;
        
        opts.init = Pcp_als;
        [P_2,output3] = cp_nc_sqp(tensor(Y),Rcp,error_bound,opts);
        
        P_2 = arrange(normalize(P_2));
        Err_(3,Rcp,krun) = norm(Y-full(P_2))/normY; %   0.0353
        R1norm_(3,Rcp,krun) = norm(P_2.lambda);%   3.8828
        RTime(3,Rcp,krun) = t;
        NoIters(3,Rcp,krun) = numel(output3.fval);
    end
end

%% Compare approximation errors 
Err_(Err_==0) = nan;

close all
fig = figure(1);
clf

h = semilogy(nanmin(Err_,[],3)');
set(h,'Linewidth',4)
set(h(2),'LineStyle',':','Linewidth',4)
set(h(3),'LineStyle','-.','Linewidth',4)


algs = {'EPC-TT2CPD' 'EPC-SVD' 'EPC-ALS'};
legend(h,algs) 
set(gca,'fontsize',15)
axis tight


xlabel('Estimated Rank')
ylabel('Relative Error')
xlim(min(15,max(1,xlim)))

% fname = sprintf('fig_hilbert_I%dN%d_err_epc',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')
% 

%% Norm of rank-1 tensors
% close all
fig = figure(2);
clf

h = semilogy(nanmean(R1norm_,3)');
set(h,'Linewidth',4)
set(h(2),'LineStyle',':','Linewidth',4)
set(h(3),'LineStyle','-.','Linewidth',4)


algs = {'EPC-TT2K' 'EPC-SVD' 'EPC-ALS'}
legend(h,algs) 
set(gca,'fontsize',15)
xlim(max(2,xlim))

xlabel('Estimated Rank')
ylabel('Rank-1 Norm')
% 
% fname = sprintf('fig_hilbert_I%dN%d_r1norm_epc',I(1),N);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')



% %%
% close all
% fig = figure(1);
% clf
% clear h
% h(1)= semilogy(output_b_ipt.fval');
% hold on
% h(2) = semilogy(output2.fval');
% set(h,'Linewidth',3)
% set(h(1),'LineStyle','--')
% set(h(2),'LineStyle',':','Linewidth',4)
% 
% algs = {'TTCP-EPC' 'SVD-EPC'}
% legend(h,algs,'location','northwest')
% % delete(h(3))
% set(gca,'fontsize',15)
% 
% xlabel('Iteration')
% ylabel('Rank-1 Norm')
