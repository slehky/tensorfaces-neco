% ----------------------- Subfunctions ----------------------------------------
function [ prox, x ] = msmooth_dual( objectiveF, mu_i, x0, ATz )
% Adding 0 to ATz will destroy the sparsity
ATz = reshape(ATz,size(x0));
if (isscalar(x0) && x0 == 0) || numel(x0) == 0 || nnz(x0) == 0
    [ v, x ] = objectiveF( mu_i * ATz, mu_i );
else
    [ v, x ] = objectiveF( x0 + mu_i * ATz, mu_i );
end
prox = tfocs_dot( ATz, x ) - v - (0.5/mu_i) * tfocs_normsq( x - x0 );
prox = -prox;
x = -x;