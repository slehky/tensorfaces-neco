% Demo for Tucker-CrNc algorithmm using different methods to select the
% step-size: the Barzilai-Borwein method (BB), or roots of a polynomial.
% 
%  This demo is a part of the TENSORBOX, 2014.
% 
clear all
%% Generate random tensor 
N = 3;                  % tensor order 
I = 100* ones(1,N);      % tensor size
Y = rand(I);
Y = tensor(Y);

R = 20;                  % Rank of the Tucker decomposition to decompose Y

% Generate initial point
Uinit = arrayfun(@(x) orth(randn(x,R)),I,'uni',0);

%% HOOI (ALS algorithm)
try
opts = struct('printitn',1,'init',{Uinit},'tol',1e-6,'maxiters',100);
tic
[T,~,out] = tucker_als(Y,R,opts);
t = toc;
catch end

%% Crank-nichol Tucker
opts = tucker_crnc;
opts.maxiters = 300;
opts.tol = 1e-6;
opts.init = Uinit;%'nvecs';
opts.crnc_param.nt = 5;
opts.crnc_param.maxiters = 20;
opts.crnc_param.stepsize = 'bb';
tic
[T2,~,out2] = tucker_crnc(Y,R,opts);
t2 = toc;


% %% Crank-nichol Tucker
% opts.crnc_param.stepsize = 'poly8';
% tic
% [T3,~,out3] = tucker_crnc(Y,R,opts);
% t3 = toc;

%% Crank-nichol Tucker
opts.crnc_param.stepsize = 'combine';
tic
[T4,~,out4] = tucker_crnc(Y,R,opts);
t4 = toc;


%%
figure(1); clf; hold on

plot(out,'r','linewidth',2)
plot(out2.Error,'k','linewidth',2)
% plot(real(out3.Error),'m','linewidth',2)
plot(real(out4.Error),'g','linewidth',2)
set(gca,'yscale','log')
set(gca,'xscale','log')
axis tight

legend({'HOOI' 'CrNc-BB'  'CrNc-comb'})

fprintf('Alg      Approx Error  Exec time (s) \n');
fprintf('HOOI     %d, %d\nCrNc-BB  %d, %d\nCrNc-cmb %d, %d\n',...
    [out(end) t out2.Error(end) t2   out4.Error(end) t4])
