% This demo compares intialization methods used in algorithms for Nonnegative
% Tucker Decompositions
%
% 1- 'nvec' :  SVD-based initialization using leading singular vectors of
%              mode-n unfoldings 
% 2- 'nmfs' :  the initialization method is similar to that based on SVD.
%              Nonnegative matrix factorizations sequentially factorize the
%              mode-n matricization as in algorithms for tensor train.
%
% 3- 'random': random initialization
%
% 4- 'fiber'   select fibers from the data tensor.
% 
% 5- Multi-initialization
%
% Initializations for Nonnegative Tucker Decomposition are implemented in
% the Matlab routine ntd_init.
%
% See: ntd_init, cp_init, ntd_hals3, ntd_flm.
%
% TENSORBOX, 2013
% Phan Anh Huy,  phan@brain.riken.jp
%
clear all
N = 3;   % tensor order
I = ones(1,N) * 50; % tensor size
R = ones(1,N) * 5;   % multilinear rank of the estimated tensor

Y = tensor(rand(I));


%% Parameters for NTD algorithm
opts = ntd_lmpu;
opts.printitn = 1;
opts.alsinit = 1;
opts.tol = 1e-6;
opts.maxiters= 2000;

exectime = [];
Fit = [];
hline = [];
algs = {};

%% Decompose tensor Y using different initialiazation methods
for k = 1:5
    switch k
        case 1
            opts.init = 'nmfs';
        case 2
            opts.init = 'fiber';
        case 3
            opts.init = 'random';
        case 4
            opts.init = 'nvecs';
        case 5
            opts.init = {'nvecs' 'fiber' 'random' 'random' 'nmfs'}; % 'dtld' can also be included 
    end
    
    tic;
    [P,output] = ntd_lmpu(Y,R,opts);
    t = toc;
     
    fit = real(output.Fit(end,2));
    
    fprintf('Fit %.4f \nExecution time %.3f seconds\n',...
        fit,t);
    
    exectime(k) = t;
    algs{k} = opts.init;
    Fit{k} = output.Fit;
end

%% Visualize and Compare results

figure(1);clf; set(gca,'fontsize',16);hold on
clrorder = get(gca,'colorOrder');
for k = 1: numel(Fit)
    h(k) = plot(Fit{k}(2:end,1),1-real(Fit{k}(2:end,2)),'color',clrorder(k,:));
end
set(h,'linewidth',2)
set(gca,'yscale','log')
set(gca,'xscale','log')

xlabel('No. Iterations')
ylabel('Relative Error')    
axis tight
algs2 = algs;
algs2(cellfun(@iscell,algs)) = {'multi'};
legend(h,algs2);


[foe,fstalg] = min(exectime);
[foe,bsttalg] = max(cellfun(@(x) x(end),Fit));

cprintf('_blue','Initialization    Exec.time (sec)  Rel. Error\n');
for k = 1: numel(exectime)
    if k == fstalg
        cprintf('blue*','%-15s   %5s           %e  \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)),1-real(Fit{k}(end,2)));
    elseif k == bsttalg
        cprintf('red*','%-15s   %5s           %e  \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)),1-real(Fit{k}(end,2)));
    else
        fprintf('%-15s   %5s           %e \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)), 1-real(Fit{k}(end,2)))
    end
end
cprintf('black','')

% TENSORBOX v1. 2013