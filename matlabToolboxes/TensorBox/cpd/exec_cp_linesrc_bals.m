function [P,outputb_,Ustore] = exec_cp_linesrc_bals(T,P,Ustore,max_lambdanorm)
% This file executes the bounded CP decomposition of the tensor T with
% initial Pb
%

if nargin <4
    max_lambdanorm = 10;
end

Rx = size(P.u{1},2);
%% STAGE 2: run BALS for CPD with minimal norm of rank-1 tensors
cost_ = [];

if nargin <3 || isempty(Ustore)
    
    P = normalize(P);
    U = P.U;
    lambda  = P.lambda;
    U{1} = U{1}*diag(lambda);
    
    
    delta = max(0,norm(T(:))^2 + norm(P)^2 - 2*innerprod(tensor(T),P));
    delta = sqrt(delta);
    
    opts = cp_anc;
    opts.maxiters = 3000;
    opts.printitn = 1;
    opts.linesearch = 0;
    opts.tol = 1e-9;
    
    Nstores = 5;
    Ustore = cell(Nstores,1);
    
    
    for k = 1:Nstores
        
        opts.init = U;
        opts.maxiters = 1;
        %[P,out_dr] = cp_fastals(tensor(T),Rx,opts);
        [P,outputb] = cp_anc(tensor(T),Rx,delta,opts);
        
        
        P = normalize(P);
        U = P.U;
        lambda  = P.lambda;
        U{1} = U{1}*diag(lambda);
        %     for n = 1:N
        %         U{n} = bsxfun(@times,U{n},(lambda.^(1/N))');
        %     end
        
        Ustore{Nstores-k+1} = U;
        
        cost_ = [cost_ ; norm(lambda)];
    end
end

normX = norm(tensor(T));

%  LINESEARCH + BALS
P = ktensor(Ustore{1});
delta = normX^2 + norm(P)^2 - 2 * real(innerprod(tensor(T),P));
delta = sqrt(delta);
outputb_ = [];
delta_ =[];


for iter = 1:4000
    Ud = cellfun(@(x,y) x-y,Ustore{1},Ustore{2},'uni',0);
    Ud2 = cellfun(@(x,y) x-y,Ustore{1},Ustore{3},'uni',0);
    
    %[U,t_sel] = cp_bound_wlinesearch(T,Ustore{1},Ud,delta,normX);
    [U,t_sel] = cp_bound_linesearch2(T,Ustore{1},Ud,Ud2,delta*1.001,normX);
    
    P = ktensor(U);
    P = normalize(P);
    U = P.U;
    lambda  = P.lambda;
    U{1} = U{1}*diag(lambda);
    
    
    cost_= [cost_ ; norm(lambda)];
    cost_diff = cost_(end) - cost_(end-1);
    
    % BALS
    opts.init = U;
    opts.maxiters = 10;
    opts.printitn = 0;
    %[P,out_dr] = cp_fastals(tensor(T),Rx,opts);
    [P,outputb] = cp_anc(tensor(T),Rx,delta,opts);
    
    P = normalize(P);
    U = P.U;
    lambda  = P.lambda;
    U{1} = U{1}*diag(lambda);
    
    cost_= [cost_ ; norm(lambda)];
    cost_diff = cost_(end) - cost_(end-2);
    
    
    fprintf(' Iter %2d: ',iter);
    fprintf('|lambda|^2 = %e diff = %7.1e\n', cost_(end), cost_diff);
    
    Ustore(2:end) = Ustore(1:end-1);
    Ustore{1} = U;
    
    if isempty(outputb_)
        outputb_ = outputb;
    else
        outputb_.Fit = [outputb_.Fit; outputb.Fit];
        outputb_.cost = [outputb_.cost; outputb.cost];
        outputb_.lambda = [outputb_.lambda; outputb.lambda];
    end
    
    if (std(outputb.lambda(end,:))<1) && (norm(outputb.lambda(end,:)) < max_lambdanorm)
        break
    end
    
    if abs(cost_diff)< 1e-8
        break
    end
    
    
end
