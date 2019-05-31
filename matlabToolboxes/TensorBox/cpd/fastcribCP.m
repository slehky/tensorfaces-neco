function [criball,crlb]=fastcribCP(U,mu)
% Fast inverse of the approximate Hessian, and update step size d
% g  gradient, dv step size,
% persistent Phi  iGn C Ptr;

I = cellfun(@(x) size(x,1),U);
N = numel(U);
R = size(U{1},2);R2 = R^2;

Km = zeros(R2,N,N);      % matrix K as a partitioned matrix
Phi = zeros(N*R2,N*R2);

iGn = zeros(R,R,N);         % inverse of Gamma
C = zeros(R,R,N);   % C in (4.5), Theorem 4.1
[Ptr,Prr] = per_vectrans(R,R); % permutation matrix Appendix A


for n = 1:N
    C(:,:,n) = U{n}'*(U{n});
end

K = zeros(N*R^2);
for n = 1:N
    % Kernel matrix
    for m = n+1:N
        gamma = prod(C(:,:,[1:n-1 n+1:m-1 m+1:N]),3); % Gamma(n,m) Eq.(4.5)
        Km(:,n,m) = gamma(:); Km(:,m,n) = gamma(:);   % Eq.(4.4)
        K((n-1)*R^2+1:(n)*R^2,(m-1)*R^2+1:(m)*R^2) = Prr*diag(gamma(:));
    end
    
    if n~=N
        gamma = gamma .* C(:,:,N) +eps;   %Gamma(n,n), Eq.(4.5)
    else
        gamma = prod(C(:,:,1:N-1),3)+eps; %Gamma(n,n), Eq.(4.5)
    end
    iGn(:,:,n) = inv(gamma+mu*eye(R));    % Eq.(4.20)
    
    % Phi = ZiGZ * K in Eq.(4.36)
    ZiGZ = kron(iGn(:,:,n),C(:,:,n)); ZiGZ = ZiGZ(:,Ptr);% Eq.(4.22)
    for m = [1:n-1 n+1:N]
        Phi((n-1)*R2+1:n*R2,(m-1)*R2+1:m*R2) = bsxfun(@times,ZiGZ,Km(:,n,m).'); % Eq.(4.36)
    end
end
Phi(1:N*R2+1:end) = 1;                    % Eq.(4.36)

%%
criball=zeros(1,R);
crlb=zeros(1,R);
kPhi = K/Phi;
for n = 1:1
    L = kron(iGn(:,:,n),U{n});
    iH = kron(iGn(:,:,n),eye(I(n))) - L * kPhi(1+(n-1)*R^2:n*R^2,1+(n-1)*R^2:n*R^2) * L';
    
    for r=1:R
        Pa1=eye(I(n))-U{n}(:,r)*U{n}(:,r)'/sum(U{n}(:,r).^2);
        criball(n,r) = trace(Pa1*iH(1+(r-1)*I(n):r*I(n),1+(r-1)*I(n):r*I(n)))/sum(U{n}(:,r).^2);
        crlb(n,r) = trace(iH(1+(r-1)*I(n):r*I(n),1+(r-1)*I(n):r*I(n)));
    end
end



end
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Perm,P] = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
Perm = reshape(1:m*n,[],n)'; Perm = Perm(:);
P = speye(m*n); P = P(Perm,:);
end
