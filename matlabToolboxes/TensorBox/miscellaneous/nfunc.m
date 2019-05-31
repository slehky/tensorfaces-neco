function [c,ceq] = nfunc(x,rho,I,crho)
U = vec2fac(x,I);P = ktensor(U);P = arrange(P);P = fixsigns(P);
U = P.U; U{1} = U{1} * diag(P.lambda);
x = fac2vec(U);
c = rho(x) - crho;
ceq = [];
