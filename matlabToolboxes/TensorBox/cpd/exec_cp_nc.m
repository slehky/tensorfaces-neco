function [Pb,outputb_] = exec_cp_bals(T,Pb,max_lambdanorm,maxiters)
% This file executes the norm reservation correction method for CPD of the
% tensor T with initial Pb
%

if nargin <3
    max_lambdanorm = 10;
end
if nargin < 4
    maxiters = 10;
end
Rx = size(Pb.u{1},2);

%% STAGE 2: run BALS for CPD with minimal norm of rank-1 tensors

% compute bound of the approximation error
delta = max(0,norm(T(:))^2 + norm(Pb)^2 - 2*innerprod(tensor(T),Pb));
delta = sqrt(delta);
% delta = 1.1*delta;

opts = cp_anc;
opts.maxiters = 3000;
opts.printitn = 1;
opts.linesearch = 0;
opts.tol = 1e-8;
outputb_ = [];


for kiter = 1:maxiters
    opts.init = Pb;
    [Pb,outputb] = cp_anc(tensor(T),Rx,delta,opts);
    
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
    if abs(outputb_.cost(end)-outputb_.cost(end))< 1e-10*outputb_.cost(end-1)
        delta = delta*1.05;
    end
end
end