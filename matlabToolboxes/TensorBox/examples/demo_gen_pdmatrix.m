% demo_gen_corrmat
% This demo generates a nonnegative correlation matrix C of size R x R with c(i,i) = 1
% and c(i,j) in a specific range [a,b]
clear all

%%
R = 30;  % size of correlation matrix
% c_range = sort(rand(1,2));
c_range = [0.2 .8]; % change here for other ranges

% Generate matrix U
U = gen_matrix(R,R,c_range);

% Correlation matrix of U
C = U'*U;

% check diagonal of C
% norm(diag(C) - 1)

coeffs = C;
coeffs(1:R+1:end) = []; % remove diagonal coeffients.

figure(1); clf;set(gca,'fontsize',18);
corr_bin = cos((acos(c_range(1)):-1/180*pi:acos(c_range(2))));
hist(coeffs,corr_bin)
hb(1) = line([c_range(1) c_range(1)], ylim,'linestyle',':','color','red','linewidth',2);
hb(2) = line([c_range(2) c_range(2)], ylim,'linestyle',':','color','red','linewidth',2);
axis auto
xlabel('Corr. pdf')
legend(hb(1),'Bound');