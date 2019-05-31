% This example illustrates how to determine rank of a CP decomposition,
% i.e. select the suitable model for CP decomposition.
% 
% The selection method is based on difference of fits achieved by CPD with
% different ranks. This example uses two methods: the convex hull and
% DIFFIT.
% 
% In order to apply the methods, 
% 
%     - Step 1: perform CPDs of the data with ranks varying in [Rmin, Rmax]. 
%     For CPD, there is only one rank interval, but for the Tucker
%     decomsposition, there may have several different intervals for
%     different modes. 
%     For a specific rank, one may need to perform CPDs with different
%     initial values, and select the best run which yields the highest fit. 
%   
%     - Step 2:  with given fits, the ConvexHull or Diffit will suggest
%     suitable ranks.
% 
% Tensorbox, 2012
% Phan Anh-Huy.

%% Generate synthetic tensor (data) of order 3 and rank R = 3

clear all
N = 3;              % Tensor order
I = ones(1,N)*10;   % Tensor size
Rtrue = 3;              % Tensor rank
 
% Generate factor matrices A{n} of the tensor.
A = cell(N,1);
for n = 1:N
    A{n} = randn(I(n),Rtrue);
end
  
Y = ktensor(A);

% Add noise to the data 
SNR = 30; % Noise level
sig2 = norm(Y)./10.^(SNR/20)/sqrt(prod(I));
if ~isinf(SNR)
    Y = full(Y);
    Y = Y + sig2 * randn(I);
end

normY = norm(Y);


%% CPD with different ranks 
N = ndims(Y);
tol = 1e-12;
maxiters = 1000;


for R = 1:10
    fprintf('Decomposition with rank %d\n',R)
    
    % Get default parameter of the CP_fLM algorithm
    cp_param = cp_fLMa;
    
    % CPD with multiple initial points, the best initial value will be
    % selected.
    cp_param.init = {'dtld' 'nvec' 'random'};                                 
    
    ts = tic;
    [Yd,outputd] = cp_fLMa(Y,R,cp_param);
    td = toc(ts);
    
    Yd = arrange(Yd); Yd = fixsigns(Yd);
    
    % Compute the approximation error and fit
    err = sqrt(normY^2 + norm(Yd)^2 - 2 * innerprod(Y,Yd));
    fitd = 1- err/normY;
    fitR(R) = fitd;    % save fit into the array fitR.
end

%% Determine suitable ranks using the CONVEXHULL method
CPrank = [];
Hullrank = 1;Rmax = numel(fitR);
Rid2 = Hullrank:Rmax;Rid2 = Rid2(:);
fitNw = fitR(Hullrank:Rmax);
while Hullrank<numel(Rid2)
    idx = Hullrank-Rid2(1)+1;
    Rid2 = Rid2(idx:end);
    fitNw = fitNw(idx:end);
    outfit = [-ones(numel(fitNw),2) Rid2,fitNw(:)];
    Hullrank = DimSelectorCP(outfit);
%     fprintf('ConvexHull CP rank %d\n', Hullrank);
    CPrank = [CPrank; Hullrank];
end

fprintf('Convexhull: CP Rank %s\n', num2str(CPrank(:)','%d, '));

%% Determine CP ranks using the DIFFIT method
Hullrank = 1;Rmax = numel(fitR);
Rid2 = (Hullrank:Rmax);Rid2 = Rid2(:);
fitNw = fitR(Hullrank:Rmax);
while 1
    dfit = diff(fitNw);
    excid = find(dfit<0);
    Rid2(excid) = [];
    fitNw(excid) = [];
    if isempty(excid)
        break
    end
end

rdfit = dfit(1:end-1)./dfit(2:end);
% excid = rdfit>normY^2/(Rid(end)-3);
% Rid2 = Rid2(excid);
% rdfit = rdfit(excid);
[foe,DFitrank] = max(rdfit);
DFitrank = Rid2(DFitrank)+1;


fprintf('DIFFIT: CP Rank %s\n', num2str(DFitrank));

figure(1);
clf
set(gca,'fontsize',20);
h(1) = plot(fitR,'b');
hold on
hullrank=plot(CPrank,fitR(CPrank),'ro','linewidth',2,'markersize',16);

hold on
dffrank = plot(DFitrank,fitR(DFitrank),'bp','linewidth',2,'markersize',16);

xlabel('Rank'); ylabel('Fit')

legend([hullrank dffrank],{'Hull rank' ,'Diffit rank'},'location','best')


% % h(2) = plot(Rid(1:end-1),dfit/max(dfit),'r');
% % h(3) = plot(Rid(3:end),rdfit/max(rdfit),'m');