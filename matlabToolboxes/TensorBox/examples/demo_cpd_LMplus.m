% Demo for LMplus for Nonnegative tensor factorization
% Generate random tensor
N = 3; % tensor order
In = [20 10 15]; % tensor size
R = 4;% tensor rank

A = cell(1,N);
alpha = zeros(N,R);
for n = 1: N
    A{n} = full(sprand(In(n),R,0.8));
    A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));    
end

lambda = ones(R,1);
Y = ktensor(lambda,A);

% Add noise to the tensor 
SNR = 30;
if ~isinf(SNR)
    sig2 = norm(Y)/10^(SNR/20)/sqrt(prod(In));
    Y = full(Y) + sig2*rand(size(Y));
end
Y = full(Y);
 
%% LMplus
opts = ncp_fLM;
opts.printitn = 1;
opts.init = 'nvec';
opts.tol = 1e-8;
[Yh,output] = ncp_fLM(Y,R,opts);
% [Yh,output] = ncp_lm_sparse(Y,R,opts);

%% Results
figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
xlabel('Iterations'); ylabel('Relative Error')

