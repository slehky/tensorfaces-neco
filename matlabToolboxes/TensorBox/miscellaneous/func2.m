function f = func2(x,rho,I)
U = vec2fac(x,I);P = ktensor(U);P = arrange(P);P = fixsigns(P);
U = P.U; U{1} = U{1} * diag(P.lambda);
x = fac2vec(U);
f = rho(x);
