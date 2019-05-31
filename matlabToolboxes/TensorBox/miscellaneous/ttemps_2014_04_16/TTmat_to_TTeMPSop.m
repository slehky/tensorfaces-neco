function x = TTmat_to_TTeMPSop( tt )
%Convert from TT-matrix to TTeMPS_op


% cores = {};
% ps = tt.ps;
% for i = 1:tt.d
%     cores{i} = reshape( tt.core( ps(i):ps(i+1)-1 ), ...
%         [tt.r(i), tt.n(i),tt.m(i), tt.r(i+1)] );
% end
r = tt.r;n = tt.n; m = tt.m;
cores = core(tt);
for i = 1:tt.d
    cores{i} = permute(reshape(cores{i},[n(i),m(i),r(i),r(i+1)]),[3 1 2 4]);
end


x = TTeMPS_op( cores );

end
