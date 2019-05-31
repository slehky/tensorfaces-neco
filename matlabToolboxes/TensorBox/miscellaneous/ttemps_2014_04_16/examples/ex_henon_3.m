% Code to produce Figure 4.4: Three Eigenvalues for Henon-Heiles with n=50, d=10
% =========================================================================
clear all
close all

n = 9;
d = 4;
A = TTeMPS_op_NN(n, d);
p = 3;  % number of eigenvalues
r = 2;

%% Run block eigenvalue procedure:
% =========================================================================

% if ~exist('hh_3_blk.mat','file')
opts = struct( 'maxiter', 3, ...
    'maxrank', 40, ...
    'tol', 1e-8, ...
    'tolOP', 1e-3, ...
    'tolLOBPCG', 1e-6, ...
    'maxiterLOBPCG', 300, ...
    'verbose', true , ...
    'precInner', true);

rng(11)
rr = [1, r * ones(1, d-1), 1];
[X_blk, C_blk, evalue_blk, residuums_blk, micro_res_blk, objective_blk, t_blk] = block_eigenvalue( A, p, rr, opts);
save('hh_3_blk', 'X_blk', 'C_blk', 'evalue_blk', 'residuums_blk', 'micro_res_blk', 'objective_blk','t_blk');
% else
%     load('hh_3_blk.mat')
% end

% Run EVAMEn:
%% =========================================================================

% if ~exist('hh_3_evamen.mat','file')
opts = struct( 'maxiter', 3, ...
    'maxrank', 40, ...
    'maxrankRes', 2, ...
    'tol', 1e-8, ...
    'tolOP', 1e-3, ...
    'tolLOBPCG', 1e-6, ...
    'maxiterLOBPCG', 300, ...
    'verbose', true , ...
    'precInner', true);
rng(11)
rr = [1, r * ones(1, d-1), 1];
[X_evamen, C_evamen, evalue_evamen, residuums_evamen, micro_res_evamen, objective_evamen, t_evamen] = amen_eigenvalue( A, 1, p, rr, opts);
save('hh_3_evamen', 'X_evamen', 'C_evamen', 'evalue_evamen', 'residuums_evamen', 'micro_res_evamen', 'objective_evamen','t_evamen');
% else
%     load('hh_3_evamen.mat')
% end


%% Prepare data for plotting:
% =========================================================================

evalue_end = repmat(evalue_blk(:,end), [1,size(evalue_blk,2)-1]);
ev_blk = abs(evalue_blk(:,1:end-1) - evalue_end);
ev_evamen = abs(evalue_evamen(:,1:end-1) - evalue_end);


% Plot vs. Iterations
% =========================================================================
f = figure
set(0,'defaultlinelinewidth',2)
subplot(1,2,1)

semilogy( sqrt(sum(micro_res_blk.^2, 1)), '-b' )
hold on
semilogy( sqrt(sum(micro_res_evamen.^2, 1)), '-k' )

semilogy( sum(ev_blk,1), '--b' )
semilogy( sum(ev_evamen,1), '--k' )

res_blk         = sqrt(sum(micro_res_blk.^2, 1))
res_evamen  = sqrt(sum(micro_res_evamen.^2, 1))

semilogy((d-1):(d-1):length(micro_res_blk),res_blk(:,(d-1):(d-1):end),'ob')
semilogy((d-1):(d-1):length(micro_res_evamen),res_evamen(:,(d-1):(d-1):end),'ok')

semilogy((d-1):(d-1):length(ev_blk),sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy((d-1):(d-1):length(ev_evamen),sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')

h_leg = legend('Res. err., Block-ALS',...
    'Res. err. EVAMEn, local prec.',...
    'EV. err., Block-ALS',...
    'EV. err. EVAMEn, local prec.')

set(gca,'fontsize',16)
set(h_leg, 'fontsize',12)
xlabel('Microiterations')
ylabel('Residual and eigenvalue error')

% Plot vs. Time
% =========================================================================

subplot(1,2,2)
semilogy( t_blk, sqrt(sum(micro_res_blk.^2, 1)), '-b' )
hold on
semilogy( t_evamen, sqrt(sum(micro_res_evamen.^2, 1)), '-k' )

semilogy( t_blk, sum(ev_blk,1), '--b' )
semilogy( t_evamen, sum(ev_evamen,1), '--k' )

semilogy(t_blk((d-1):(d-1):end),res_blk(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen((d-1):(d-1):end),res_evamen(:,(d-1):(d-1):end),'ok')

semilogy(t_blk((d-1):(d-1):end),sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen((d-1):(d-1):end),sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')

semilogy(t_blk((d-1):(d-1):end),        res_blk(:,(d-1):(d-1):end),'ob')
semilogy(t_evamen((d-1):(d-1):end), res_evamen(:,(d-1):(d-1):end),'ok')

semilogy(t_blk((d-1):(d-1):end),        sum(ev_blk(:,(d-1):(d-1):end),1),'ob')
semilogy(t_evamen((d-1):(d-1):end), sum(ev_evamen(:,(d-1):(d-1):end),1),'ok')


h_leg = legend('Res. err., Block-ALS',...
    'Res. err. EVAMEn',...
    'EV. err., Block-ALS',...
    'EV. err. EVAMEn')
set(gca,'fontsize',16)
set(h_leg, 'fontsize',12)
xlabel('Time [s]')
ylabel('Residual and eigenvalue error')

set(f, 'Position', [0 0 1200 700])
