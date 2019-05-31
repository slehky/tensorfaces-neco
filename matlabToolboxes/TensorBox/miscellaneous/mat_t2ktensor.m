function M = mat_t2ktensor(R)
% Generate conversion matrices Mn of size Rn x R1..RN in conversion of a
% t-tensor of rank-(R1,R2,...,RN) into a k-tensor of rank R1R2...RN
% 
% G x1 U1 x2 U2 ... xN UN
%
% dvec(G) x1 (U1 M1) x2 (U2M2) ... xN (UN MN)
% 
% Tensorbox v.2014
% Phan Anh Huy
%
N = numel(R);
% CR = prod(R);
% M = cell(N,1);
% T = reshape(1:prod(R),R);
% for n = 1:N
%     Mm = kron(eye(R(n)),ones(1,CR/R(n)));
%     Mm = reshape(Mm,[R(n) R]);
%     Mm = permute(Mm,[1 n+1 2:n n+2:N+1]);
%     Mm = reshape(Mm,R(n),[]);
%     
%     M{n} = Mm;
% end

M = cell(N,1);
for n = 1:N
    M{n} = kron(kron(ones(1,prod(R(n+1:end))),eye(R(n))),ones(1,prod(R(1:n-1))));
end
