function [U,S,V] = msvd(Y)
SzY = size(Y);
mz = min(SzY); Mz = max(SzY);
if (Mz/mz>5) && (Mz >= 5e3)
    if SzY(1)>SzY(2)
        C = Y'*Y;
        [V,S] = eig(C);
        S = (sqrt(diag(S)));
        U = Y*V*diag(1./S);
    else
        C = Y*Y';
        [U,S] = eig(C);
        S = (sqrt(diag(S)));
        V = diag(1./S)*U'*Y;
    end
    S = diag(S);
else
    [U,S,V] = svd(Y,'econ');
end
end