function [x_s]= sparse_vec(x,errorGoal)
% Find a sub-set Omega of vector x such that x_Omega is sparsest while its
% approximation error is smaller than the noise level
%=============================================
%  min_omega |x_omega|_1   subject to  ||x_omega||_2^2 > epsilon
%=============================================
% Anh-Huy Phan, 2015

[n,P]=size(x);
E2 = errorGoal^2*n;

[x_st,x_ix] = sort(abs(x),'descend');
ell2_x = cumsum(x_st.^2);
E3 = ell2_x(end,:)-E2;
x_s = zeros(size(x));
for kk = 1:size(x,2)
    mix = find(ell2_x(:,kk)>= E3(kk),1);
    
    if ~isempty(mix)
        indx = x_ix(1:mix,kk);
        x_s(indx,kk) = x(indx,kk);
    end
end
