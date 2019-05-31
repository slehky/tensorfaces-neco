%% This example compares algorithms for Nonnegative Tucker decomposition.
% TENSORBOX ver.1.

clear all;
N = 3;
I = [30 40 50];  % tensor size
R = [3 4 5];    % multilinear rank of the estimated tensor

Y = tensor(rand(I));

%% Generate initial values 

opts = ntd_init;
opts.init = 'nmfs';
opts.maxiters = 5000;
opts.alsinit = 1;
opts.printitn = 1;
opts.tol = 1e-7;

[A,G] = ntd_init(Y,R,opts);

Yi = ttensor(G,A);

%% Decompose tensor Y by NTD algorithms

ntd_algs = {@ntd_hals3 @ntd_o2lb @ntd_lm @ntd_lmpu };

exectime = zeros(numel(ntd_algs),1);
Fit = cell(numel(ntd_algs),1);
%%
for k = 1:numel(ntd_algs)
    algs = ntd_algs{k};
    
    opts = algs();
    opts.init = Yi; %'nmfs';
    opts.maxiters = 5000;
    opts.alsinit = 1;
    opts.printitn = 1;
    opts.tol = 1e-8;
    
    tic;
    [T,output] = algs(Y,R,opts);
    ts = toc;
    
    fit = real(output.Fit(end,2));
    
    fprintf('%s, Fit %.4f  Exec. time %.3f seconds\n',func2str(algs),fit,ts);
    
    exectime(k) = ts;
    Fit{k} = output.Fit;
end

%% Compare cost values as functions of iterations
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

algs2 = cellfun(@func2str,ntd_algs,'uni',0);
algs2 = cellfun(@(x) x(5:end),algs2,'uni',0);
legend(h,algs2);

[foe,fstalg] = min(exectime);
[foe,bsttalg] = max(cellfun(@(x) x(end),Fit));

cprintf('_blue','Algorithm    Exec.time (sec)  Rel. Error\n');
for k = 1: numel(exectime)
    if k == fstalg
        cprintf('blue*','%-10s   %5s           %e  \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)),1-real(Fit{k}(end,2)));
    elseif k == bsttalg
        cprintf('red*','%-10s   %5s           %e  \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)),1-real(Fit{k}(end,2)));
    else
        fprintf('%-10s   %5s           %e \n',...
            algs2{k},sprintf('%-2.2f',exectime(k)), 1-real(Fit{k}(end,2)))
    end
end
cprintf('black','')
