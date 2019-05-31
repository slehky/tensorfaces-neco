% This demo shows how to use the fastALS algorithm for CANDECOMP/PARAFAC.
%
% The example decomposes a 3-way Amino acids fluorescence data, which is
% provided at http://www.models.life.ku.dk/Amino_Acid_fluo.
%
% TENSORBOX, 2012
% Phan Anh Huy,  phan@brain.riken.jp
%
% Load your data here
clear all
load amino.mat

% Force the three-way array X of size 5 x 201 x 61 to be an order-3 tensor
X = tensor(X);   

% Dimension of the tensor X should be arranged in ascending order in order
% to accelerate computation of CP gradients, i.e., 5 x 61 x 201. 
% Note that algorithms in TENSORBOX will rearrange dimensions of the
% tensor in ascending order.
X = permute(X,[1 3 2]); 

%% Get parameters of the FastALS or any other algorithms in TENSORBOX
opts = cp_fastals;      % get default setting of FastALS

% Set initialization type for factor matrices of the CP decomposition. 
opts.init = 'dtld';  % GRAM initialization. 
% For mutli-initializations, the best one will be chosen after some small
% runs. 
% opts.init = {'nvec' 'random' 'random' 'fiber' 'fiber'};
                        
opts.printitn = 5;      % print Fit values every "printitn" iterations
opts.maxiters = 100;    % No. iterations

R = 3;                  % No of columns of factor matrices

tic;
[P,output] = cp_fastals(X,R,opts); % P is a rank-R K-tensor with N factors 
                        % P.U{1},...,P.U{N} and coefficients P.lambda
t = toc;

fit = real(output.Fit(end,2));

fprintf('Final Fit %.4f \nExecution time %.3f seconds\n',fit,t);

figure(1);clf;set(gca,'fontsize',14); hold on
h = plot(output.Fit(:,1),1-output.Fit(:,2));
xlabel('Iterations')
ylabel('Relative Error')

%% Visualize the third factor matrix
U = P.U;
EmAx = 250:450;

figure(2);clf;set(gca,'fontsize',14);
hold on
plot(EmAx,U{3});

axis tight ; grid on
xlabel('Em. Wavelength/nm')
ylabel('Relative intensity')

% TENSORBOX v1. 2012