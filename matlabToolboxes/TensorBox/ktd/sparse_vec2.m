function [x_s]= sparse_vec2(x,errorGoal)
% Find a sub-set Omega of vector x such that x_Omega is sparsest while its
% approximation error is smaller than the noise level
%=============================================
%  min_omega |x_omega|_1   subject to  ||x_omega||_2^2 > epsilon
%=============================================
% Anh-Huy Phan, 2015
%
%   find largest Omega such that  
%       |x_omega|_2 <  c1 * |x_omega|_1 + c2 * noise_variance 
%
% see more Candes, Romberg and Tao, 2006.


% [n,P]=size(x);
% E2 = errorGoal*sqrt(n);
%  
% [x_st,x_ix] = sort(abs(x),'ascend');
% ell2_x = sqrt(cumsum(x_st.^2));
% ell1_x = cumsum(abs(x_st));
%  
% 
% x_s= sparse_vec(x,errorGoal);
% 
% d2 = sqrt(sum((x-x_s).^2));
% d1 = sum(abs(x-x_s));
% 
% % Find c1 and c2 such that c1*d1 + c2 *E2 - d2>0
% 
% tt = ([d1' E2*ones(size(d1))']);
% 
% c = max((tt'*tt)\(tt'*d2'),0);
% 
% d_ell = (ell2_x - c(1) * ell1_x)/c(2); % <= E2
% 
% 
% x_s = x;
% for kk = 1:size(x,2)
%     mix = find(d_ell(:,kk) < E2,1,'last');
%     if isempty(mix)
%         mix = n;
%     end
%     
%     indx = x_ix(1:mix,kk);
%     x_s(indx,kk) = 0; %;x(indx,kk);
% end


[n,P]=size(x);
E2 = errorGoal*sqrt(n);
 
[x_st,x_ix] = sort(abs(x),'ascend');
ell2_x = sqrt(cumsum(x_st.^2));
ell1_x = cumsum(abs(x_st));
 
% find the first entry indices xs_(k) such that norm(xs(1:k-1)) < E2 < norm(xs(1:k))
[mix,ic] = find(diff(ell2_x>E2)==1);

x_s = zeros(size(x));
for kk = 1:numel(ic)
    if mix(kk) ~= 0
        indx = x_ix(mix(kk)+1:end,ic(kk));
        x_s(indx,ic(kk)) = x(indx,ic(kk));
    end
end


%% Find  c1 and c2 such that d2 < c1*d1 + c2 *E2 
dx = x-x_s;
d2 = sqrt(sum((dx).^2));
d1 = sum(abs(dx));

tt = ([d1' E2*ones(size(d1))']);
c = max((tt'*tt)\(tt'*d2'),0);

%% Find the solution
d_ell = (ell2_x - c(1) * ell1_x)/c(2); % <= E2

x_s = zeros(size(x));
for kk = 1:size(x,2)
    mix = find(d_ell(:,kk) < E2,1,'last');
    
    if ~isempty(mix)
        indx = x_ix(mix+1:end,kk);
        x_s(indx,kk) = x(indx,kk);
    end
end