function [crib,iGn]=fastcribCP(varargin)
% function crib=fastcribCP(U)
% or 
% function crib=fastcribCP(C,I)
% CRIB for the first component of the first factor matrix U{1}(:,1)
%
% In order to compute CRIBs for all compoentns in all factors
%
% for n = 1:N
%     Un = U([n 1:n-1 n+1:N]);
%     for r = 1:R
%         Ur = cellfun(@(x) x(:,[r 1:r-1 r+1:end]),Un,'uni',0);
%         crib(n,r) = fastcribCP(Ur) ;
%     end
% end
%
if nargin == 1
    U = varargin{1};
    I = cellfun(@(x) size(x,1),U);
    N = numel(U);
    R = size(U{1},2);
    
    % Normalize U before computing CRIB
    lambda = ones(1,R);
    for n = 1:N
        am = sqrt(sum(U{n}.^2));
        lambda = lambda .* am;
        U{n} = bsxfun(@rdivide,U{n},am);
    end
    
    C = zeros(R,R,N);   % C in (4.5), Theorem 4.1
    for n = 1:N
        C(:,:,n) = U{n}'*(U{n});
    end
else
    C = varargin{1}; I = varargin{2};
    [R,R,N] = size(C); % correlation matrices Cn = Un'*Un
%     if nargin >= 3
%         lambda = varargin{3};
        % Check diagonal 
        lambda = ones(R,1);
        for n = 1:N
            lambdan = diag(C(:,:,n));
            C(:,:,n) = bsxfun(@rdivide,C(:,:,n),sqrt(lambdan));
            C(:,:,n) = bsxfun(@rdivide,C(:,:,n),sqrt(lambdan'));
            lambda = lambda.*lambdan;
        end
        lambda = sqrt(lambda);
%     else
%         lambda = ones(1,N);
%     end
end
R2 = R^2;

flag = false;
for n = 1:N
    Gamma = prod(C(:,:,[1:n-1 n+1:end]),3);
    % Check collinearity of factor matrices U
    if (cond(Gamma) > 1e5)
        warning('CRIB:singularMatrix','High collinearity in %s{n#%d}',inputname(1),n);
        flag = true;
    end
end
if flag
    warning('CPD:collinearity','CP-Decomposition is not stable. Try PARALIN or CONFAC.');
end

Km = zeros(R2,N,N);      % matrix K as a partitioned matrix
Phi = zeros(N*R2,N*R2);

iGn = zeros(R,R,N);         % inverse of Gamma
[Ptr,Prr] = per_vectrans(R,R); % permutation matrix Appendix A


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
    iGn(:,:,n) = inv(gamma);    % Eq.(4.20)
    
    % Phi = ZiGZ * K in Eq.(4.36)
    if n == 1
        ZiGZ = kron(iGn(:,:,n),C(:,:,n)); 
    else
        ZiGZ = kron(iGn(:,:,n),C(:,:,n) - 1/C(1,1,n) * C(:,1,n) * C(1,:,n));
    end
    ZiGZ = ZiGZ(:,Ptr);% Eq.(4.22)
    for m = [1:n-1 n+1:N]
        Phi((n-1)*R2+1:n*R2,(m-1)*R2+1:m*R2) = bsxfun(@times,ZiGZ,Km(:,n,m).'); % Eq.(4.36)
    end
end
Phi(1:N*R2+1:end) = 1;                    % Eq.(4.36)

B = K/Phi;

B0 = B(1:R^2,1:R^2);
X1 = C(:,:,1) - 1/C(1,1,1) * C(:,1,1) * C(1,:,1);
crib = iGn(1,1,1) * (I(1) -1) - trace(B0 * kron(iGn(1,:,1).' * iGn(1,:,1), X1));

crib = crib/lambda(1)^2;
end
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Perm,P] = per_vectrans(m,n)
% vec(X_mn^T) = P vec(X)
Perm = reshape(1:m*n,[],n)'; Perm = Perm(:);
P = speye(m*n); P = P(Perm,:);
end
