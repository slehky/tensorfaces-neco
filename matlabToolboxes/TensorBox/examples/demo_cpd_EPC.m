% This example illustrates the Error preserving correction method for CPD
% 
% 
% The considered tensor is of size 4 x 4 x 4 and has rank-5.
% The first 4 column components are highly collinear.
% 
% The tensor is first decomposed using ALS in 10 (50) iterations to give initial
% values.
%
% The decomposition is then continue with and without the Error Preserving
% Correction (EPC) method 
%     
% The result will show that estimated tensors using the traditional
% algorithms for CPD will have high norms of rank-1 tensors. 
% The EPC method can replace the tensor with high norm by another tensor
% with smaller norm of rank-1 tensors.
% This helps CPD easier to fit the data.
%
% TENSORBOX
%
clear all
warning off
N = 3;   % tensor order
R = 5;   % tensor rank
I = (R-1)*ones(1,N); % tensor size

% Generate a Kruskal tensor from random factor matrices which have
% collinearity coefficients distributed in specific ranges
c = [0.95 0.95;  0.95 0.95;  0.95 0.95]+.04;

A = cell(N,1);
for n = 1:N
    if R<I(n)
        A{n} = gen_matrix(I(n),R,c(n,:));
    else
        An =[];
        for kn = 1:ceil(R/I(n))
            rn = min(I(n),R-(kn-1)*I(n));
            if rn>1
                An = [An gen_matrix(I(n),rn,c(n,:))];
            else
                v = randn(I(n),rn);
                An = [An v/norm(v)];
            end
        end
        A{n} = An;
    end
end
 
% Add Gaussian noise
% lambda = 1+rand(R,1);
lambda = ones(R,1);
Y = ktensor(lambda,A(:));  % a Kruskal tensor of rank-R

Y = normalize(Y);
rank1normbound = norm(Y.lambda);

Y = tensor(Y);
Rx = R;
 
%% STAGE 1: run ALS in a few iterations (10, 50  iterations)
% Run the CPD decompostion uses [I 1] as initial factor matrices
Pi = ktensor({[eye(I(1)) ones(I(1),1)] ;[eye(I(1)) ones(I(1),1)] ;[eye(I(1)) ones(I(1),1)]});

opts = cp_fastals;
%opts.init = { 'dtld' 'nvec'  'nvec' 'rand'};
% opts.init = repmat({'rand'},1,10);
opts.init = Pi;
opts.printitn = 1;
opts.maxiters = 50;
opts.linesearch = 0;
% opts.maxiters = 50;
opts.TraceRank1Norm = true;

opts.tol = 1e-9;


[P_als,out_als] = cp_fastals(tensor(Y),Rx,opts);
 
% result in the first stage
U = cellfun(@(x) bsxfun(@times,x,P_als.lambda'.^(1/3)),P_als.u,'uni',0);
P_als = ktensor(U);
 
%% STAGE 2: CPD using P_als as initial values

algorithms = {@cp_boundlda_sqp % SQP algorithm for bounded CPD    min |Y-Yx|_F^2  s.t.  |norm_rank-1_tensors|_F^2 <= delta
              @cp_boundlda_ipm % Iterior Point method      min |Y-Yx|_F^2  s.t.  |norm_rank-1_tensors|_F^2 <= delta
              @cp_boundlda_als % ALS for EPC               min |Y-Yx|_F^2  s.t.  |norm_rank-1_tensors|_F^2 <= delta 
              @cp_fLMa         % LM for CPD                min |Y-Yx|_F^2 
              @cp_fastals};    % ALS for CPD               min |Y-Yx|_F^2  
algs_name = {'BSQP' 'BITP' 'BALS' 'fLM' 'fALS'};

% delta = 50;

Output = cell(numel(algorithms),1);
Rank1Norm = cell(numel(algorithms),1);

%
for ka = 1:numel(algorithms)
    if (ka == 1) || (ka == 2)
        delta =  sqrt(1.1)*rank1normbound;
    elseif ka == 3
        delta =  sqrt(1.0)*rank1normbound;
    end
    cp_alg = algorithms{ka};
     
    opts = cp_alg();
    opts.init = P_als;
    opts.linesearch = 0;
    opts.alsinit = 0;
    opts.tol = 1e-10;
    opts.maxiters = 5000;
    opts.ineqnonlin = true;
    % opts.Algorithm = 'sqp'
    opts.Correct_Hessian = true; % good with the equality constraint
    % opts.bound_ell2_constraint = true;  \|\theta\|_F^2< xi
    opts.bound_ell2_constraint = false;
    opts.printitn = 1;
    opts.TraceRank1Norm = true; 
    
    try
        [Pbx,outputx] = cp_alg(tensor(Y),Rx,delta,opts);
        
        if isfield(outputx,'Fit')
            Output{ka} = (1-real(outputx.Fit(:,2))).^2;
        else
            Output{ka} = 2*outputx.fval;
        end
    catch
        %% fLM
        [Pbx,outputx] = cp_alg(tensor(Y),Rx,opts);
        Output{ka} = (1-real(outputx.Fit(:,2))).^2;
    end
    if isfield(outputx,'Rank1Norm')
        Rank1Norm{ka} = (sum(outputx.Rank1Norm.^2,2));
    end 
end

% Display the approximation errors (cost function values)
figure(2); clf; hold on
clear h;
for k = 1:numel(Output)
    h(k) = loglog(sqrt(Output{k}));
end
title('CPD without Error Preservation Correction on initial values')
legend(h,algs_name)
set(h,'linewidth',3)
set(gca,'YScale','log')
set(gca,'xScale','log')
xlabel('Iterations')
ylabel('Relative Error')

% -10*log10(cellfun(@(x) x(end),Output)')

%% STAGE 3a: Before runnning CPD, perform the Error Preserving Correction 
% to norm of rank-1 tensors 

opts = cp_anc;
opts.maxiters = 2000;
opts.printitn = 1;
opts.linesearch = 0;
opts.tol = 1e-6;

opts.init = P_als;
delta = norm(Y(:))^2 + norm(P_als)^2 - 2*innerprod(Y,P_als);
delta = sqrt(delta);

[P_als_bals,output_als_bals] = cp_anc(Y,R,1.1*delta,opts);

%% STAGE 3b: CPD of the tensor Y using results after the EPC
algorithms = {@cp_boundlda_sqp @cp_boundlda_ipm @cp_boundlda_als @cp_fLMa @cp_fastals};

delta = 1.01*rank1normbound;

Output_normcorrection = cell(numel(algorithms),1);
Rank1Norm_normcorrection = cell(numel(algorithms),1);

for ka = 1:numel(algorithms)
    cp_alg = algorithms{ka};
    
    opts = cp_alg();
    opts.init = P_als_bals; % set initial value here 
    opts.alsinit = 0;
    opts.tol = 1e-10;
    opts.maxiters = 5000;
    opts.linesearch = 0;
    opts.ineqnonlin = true;
    % opts.Algorithm = 'sqp'
    opts.Correct_Hessian = true; % good with the equality constraint
    % opts.bound_ell2_constraint = true;  \|\theta\|_F^2< xi
    opts.bound_ell2_constraint = false;
    opts.printitn = 1;
    opts.TraceRank1Norm = true; 
    
    try
        [Pbx,outputx] = cp_alg(tensor(Y),Rx,delta,opts);
        
        if isfield(outputx,'Fit')
            Output_normcorrection{ka} = (1-real(outputx.Fit(:,2))).^2;
        else
            Output_normcorrection{ka} = 2*outputx.fval;
        end
    catch
        %% fLM
        [Pbx,outputx] = cp_alg(tensor(Y),Rx,opts);
        Output_normcorrection{ka} = (1-real(outputx.Fit(:,2))).^2;
    end
    if isfield(outputx,'Rank1Norm')
        Rank1Norm_normcorrection{ka} = (sum(outputx.Rank1Norm.^2,2));
    end
end

figure(3); clf; hold on
clear h;
for k = 1:numel(Output_normcorrection)
    h(k) = loglog(sqrt(Output_normcorrection{k}));
end
legend(h,algs_name)
set(h,'linewidth',3)
set(gca,'YScale','log')
set(gca,'xScale','log')
xlabel('Iterations')
ylabel('Relative Error')
title('CPD with Error Preservation Correction on initial values')
 

% -10*log10(cellfun(@(x) x(end),Output_normcorrection)')

%% Compare Frobenius norm of rank-1 tensors with and without EPC

fig = figure(3);clf; hold on
clear h;

% Norm of rank-1 tensor components estimated by ALS 
% ALS (initialization) -> ALS
rank1_norm1 = sum(out_als.Rank1Norm.^2,2); n1 = numel(rank1_norm1);
rank1_norm2 = Rank1Norm{end};n2 = numel(rank1_norm2);
hals = plot([rank1_norm1; rank1_norm2]); hold on
set(hals,'linewidth',3)

% Norm of rank-1 tensor components estimated by ALS after EPC
% ALS (initialization)-> EPC -> ALS
rank1_norm2 = output_als_bals.cost;n2 = numel(rank1_norm2);
rank1_norm3 = Rank1Norm_normcorrection{end};n3 = numel(rank1_norm3);

clear h2
h2(1) = plot(n1:n1+n2,[rank1_norm1(end) ; rank1_norm2]); hold on
h2(2) = plot(n1+n2:n1+n2+n3,[rank1_norm2(end) ; rank1_norm3]); hold on
set(h2,'linewidth',3)
set(h2,'LineStyle','--')

set(gca,'yScale','log')
set(gca,'xScale','log')
grid on
set(gca,'fontsize',16)
xlabel('Iterations')
ylabel('Norm of Rank-1 Tensors')

legend([hals h2],{'ALS' 'ALS->EPC->ALS'},'location','best')
 
% fname = sprintf('fig_R%dI%d_snr%d_lambda',R,I(1),SNR);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc')

%% Compare relative approximation errors for ALS vs ALS with correction 

fig = figure(4); clf; hold on
clear h;

% ALS 
h(1) = loglog([1-real(out_als.Fit(:,2)); sqrt(Output{end})]);
% ALS + EPC 
h(2) = loglog([1-real(out_als.Fit(:,2)); sqrt(Output_normcorrection{end})]);

set(h(2),'LineStyle','--')
set(h,'linewidth',3)
set(gca,'YScale','log')
set(gca,'xScale','log')
grid on
set(gca,'fontsize',16)
xlabel('Iterations')
ylabel('Relative Error')

legend(h,{'ALS' 'ALS+Norm correction'},'location','best')

% fname = sprintf('fig_R%dI%d_snr%d_relerror_ALS',R,I(1),SNR);
% saveas(fig,fname,'fig')
% print(fig,fname,'-depsc') 