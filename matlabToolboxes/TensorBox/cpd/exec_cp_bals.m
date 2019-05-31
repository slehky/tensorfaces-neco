function [Pb,outputb_] = exec_cp_bals(T,Pb,max_lambdanorm,maxiters,nc_alg)
% This file executes the bounded CP decomposition of the tensor T with
% initial Pb
%

if nargin <3
    max_lambdanorm = 10;
end
if nargin < 4
    maxiters = 10;
end

if nargin<5
    nc_alg = @cp_anc;
end
Rx = size(Pb.u{1},2);
%% STAGE 2: run ANC for CPD with minimal norm of rank-1 tensors

% compute bound of the approximation error
delta = max(0,norm(T(:))^2 + norm(Pb)^2 - 2*innerprod(tensor(T),Pb));
delta = sqrt(delta);
% delta = 1.1*delta;

opts = nc_alg();
opts.maxiters = 3000;
opts.printitn = 1;
opts.linesearch = 0;
opts.tol = 1e-8;
outputb_ = [];


for kiter = 1:maxiters
    opts.init = Pb;
    [Pb,outputb] = nc_alg(tensor(T),Rx,delta,opts);
    
    if isempty(outputb_)
        outputb_ = outputb;
    else
        outputb_.Fit = [outputb_.Fit; outputb.Fit];
        outputb_.cost = [outputb_.cost; outputb.cost];
        outputb_.lambda = [outputb_.lambda; outputb.lambda];
    end
    
    if (std(outputb.lambda(end,:))<1) && (norm(outputb.lambda(end,:)) <= max_lambdanorm)
        break
    end
    if abs(outputb_.cost(end)-outputb_.cost(end))< 1e-10*outputb_.cost(end-1)
        delta = delta*1.05;
    end
end
end