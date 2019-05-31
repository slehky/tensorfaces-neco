function [criball,crbf,H,Hi2] = cribCP1(U,Weights)
%
% Computes Cramer-Rao- induced lower bound on mean square angular error in
% estimating the first colunm of the first factor matrix
% from a noisy observation of a four-way tensor I x_1 A x_2 B x_3 C x_4 D
% The noise  is assumed to be i.i.d. Gaussian of a variance sigma^2, the
% output CRB is proportional to sigma^2, and without any loss in generality
% it is assumed to be  1.
%
% U: cell array of factor matrices
% 
%
% Ref:
% P. Tichavsk?, A.-H. Phan, and Z. Koldovsk?, ?Cram?r-Rao-induced bounds
% for CANDECOMP/PARAFAC tensor decomposition,? IEEE Trans. Signal Process.,
% vol. 61, no. 8, pp. 1986?1997, 2013. 
% 

if nargin <2
    Weights = [];
end

I = cellfun(@(x) size(x,1),U);I = I(:)';
r = size(U{1},2);
Isum =sum(I);
N = numel(U);
UtU = zeros(r,r,N);
for n = 1:N
    UtU(:,:,n) = U{n}'* U{n};
end
for n = 1:N
    mu(:,:,n) = prod(UtU(:,:,[1:n-1 n+1:N]),3);
end
Ir=eye(r);
cI = cumsum([0 I]);
%H = [];
H=zeros(r*Isum,r*Isum);  %%% will be the Hessian matrix

%%
if isempty(Weights)
    for j=1:r
        for k=1:r
            Hjk = zeros(sum(I),sum(I));
            for n1 = 1:N
                for n2 = 1:N
                    if n2 == n1
                        temp2 = mu(j,k,n1)*eye(I(n1));
                    else
                        temp2 = prod(UtU(j,k,setdiff(1:N,[n1,n2]))) * U{n1}(:,j)*U{n2}(:,k)';
                    end
                    Hjk(cI(n1)+1:cI(n1)+I(n1),cI(n2)+1:cI(n2)+I(n2)) = temp2;
                end
            end
            H(Isum*(k-1)+1:Isum*k,Isum*(j-1)+1:Isum*j) = Hjk;
        end
    end
    
    %% Fast construction of Hessian for missing data
else
    
    R = r;
    cIR = [0 cumsum(I)*R];
    Pp = [];G2 =[];
    for n = 1:N
        Pnr = per_vectrans(R,I(n));
        Pp = blkdiag(Pp,Pnr);
        G2 = blkdiag(G2,Pnr * kron(U{n},U{n}'));
    end
    Psum = per_vectrans(R,Isum);
    UU = cell2mat(U(:));
    D = kron(eye(Isum),ones(R));
    F = kron(UU,UU') * Psum ;
    Hu = Pp* (D + F) * Pp' - G2;
    
    Wu = zeros(Isum*R);
    %     Weights = tensor(Weights);
    %     for kr = 1:R
    %         for ks = kr:R
    %             urs = cellfun(@(x) x(:,kr).*x(:, ks),U,'uni',0);
    %             for n = 1:N
    %                 for m = n:N
    %                     Wt2 = ttv(Weights,urs,setdiff(1:N,[n,m]));
    %                     Wt2 = double(Wt2);
    %                     if m<n
    %                         Wt2 = Wt2';
    %                     end
    %                     if n==m
    %                         Wnmrs = diag(Wt2);
    %                     else
    %                         Wnmrs = Wt2;
    %                     end
    %                     Wu(cIR(n)+(kr-1)*I(n)+1:cIR(n)+(kr)*I(n),cIR(m)+(ks-1)*I(m)+1:cIR(m)+(ks)*I(m)) = Wnmrs;
    %                     Wu(cIR(n)+(ks-1)*I(n)+1:cIR(n)+(ks)*I(n),cIR(m)+(kr-1)*I(m)+1:cIR(m)+(kr)*I(m)) = Wnmrs;
    %
    %                     Wu(cIR(m)+(kr-1)*I(m)+1:cIR(m)+(kr)*I(m),cIR(n)+(ks-1)*I(n)+1:cIR(n)+(ks)*I(n)) = Wnmrs';
    %                     Wu(cIR(m)+(ks-1)*I(m)+1:cIR(m)+(ks)*I(m),cIR(n)+(kr-1)*I(n)+1:cIR(n)+(kr)*I(n)) = Wnmrs';
    %                 end
    %             end
    %         end
    %     end
    
    
    for r = 1:R
        Rr = R-r+1;
        urs = cellfun(@(x) bsxfun(@times,x(:,r:end),x(:,r)),U,'uni',0);
        G = cp_gradients(Weights,urs);
        
        for n = 1:N
            m = n;
            Wm = G{n};
            for is = 1:Rr
                s = r+is-1;
                Wnmrs = diag(Wm(:,is));
                
                Wu(cIR(n)+(r-1)*I(n)+1:cIR(n)+(r)*I(n),cIR(m)+(s-1)*I(m)+1:cIR(m)+(s)*I(m)) = Wnmrs';
                Wu(cIR(n)+(s-1)*I(n)+1:cIR(n)+(s)*I(n),cIR(m)+(r-1)*I(m)+1:cIR(m)+(r)*I(m)) = Wnmrs';
            end
            %         norm(Wu-Wu2,'fro')
        end
        
        for n = N:-1:1
            % Left projection
            if n<N
                if n == N-1
                    Ukrp_left = urs{end};
                else
                    Ukrp_left = khatrirao(Ukrp_left,urs{n+1});
                end
                Wn = reshape(Weights,prod(I(1:n)),[]);
                Wn = Wn * Ukrp_left; % I1 x ... x In x R
            else
                Wn = double(Weights); % I1 x ... x In
            end
            
            for m = 1:n-1
                % right projection
                if m>1
                    if m == 2
                        Ukrp_right = urs{1};
                    else
                        Ukrp_right = khatrirao(urs{m-1},Ukrp_right);
                    end
                    %Ukrp_right = khatrirao(urs(m-1:-1:1));
                    
                    if n == N
                        Wm = reshape(Wn,prod(I(1:m-1)),[]);
                        Wm = Wm'*Ukrp_right;  % Im x ... x In x R
                    else
                        Wm = reshape(Wn,prod(I(1:m-1)),[],I(n),Rr);
                        Wm = bsxfun(@times,Wm,reshape(Ukrp_right,prod(I(1:m-1)),1,1,Rr));
                        Wm = sum(Wm,1); % Im x ... x In x R
                    end
                else
                    Wm = Wn; % Else Wm : Im x ... x In
                end
                
                if n>m+1
                    Ukrp = khatrirao(urs(n-1:-1:m+1));
                    
                    if numel(Wm) == prod(I(m:n))
                        Wm = reshape(Wm,I(m), [],I(n));
                        Wm = permute(Wm,[1 3 2]);
                        Wm = reshape(Wm,I(n)*I(m),[]);
                        Wm = Wm * Ukrp;
                    else
                        Wm = reshape(Wm,I(m), [],I(n),Rr);
                        Wm = bsxfun(@times,Wm,reshape(Ukrp,1,prod(I(m+1:n-1)),1,Rr));
                        Wm = squeeze(sum(Wm,2)); % Im x ... x I x R
                    end
                end
                Wm = reshape(Wm,I(m),I(n),[]); % Im x ... x In x R
                
                for is = 1:R-r+1
                    s = r+is-1;
                    
                    if n==m
                        Wnmrs = diag(Wm(:,1,is));
                    else
                        Wnmrs = Wm(:,:,is);
                    end
                    Wu(cIR(n)+(r-1)*I(n)+1:cIR(n)+(r)*I(n),cIR(m)+(s-1)*I(m)+1:cIR(m)+(s)*I(m)) = Wnmrs';
                    Wu(cIR(n)+(s-1)*I(n)+1:cIR(n)+(s)*I(n),cIR(m)+(r-1)*I(m)+1:cIR(m)+(r)*I(m)) = Wnmrs';
                    
                    Wu(cIR(m)+(r-1)*I(m)+1:cIR(m)+(r)*I(m),cIR(n)+(s-1)*I(n)+1:cIR(n)+(s)*I(n)) = Wnmrs;
                    Wu(cIR(m)+(s-1)*I(m)+1:cIR(m)+(s)*I(m),cIR(n)+(r-1)*I(n)+1:cIR(n)+(r)*I(n)) = Wnmrs;
                end
            end
            %norm(Wu-Wu2,'fro')
        end
    end
    
    H = Hu.*Wu;
    H = Psum*Pp' * H* Pp*Psum' ;
    %     norm( Psum*Pp' * H* Pp*Psum' - H,'fro')
    
    
    %     % Construct Jacobian and Hessian
    %     W1 = diag(Weights(:));
    %     J = [];
    %     for n = 1: N
    %         Pn = permute_vec(I,n);
    %         Jn = kron(khatrirao(U(setdiff(1:N,n)),'r'),eye(I(n)));
    %         J = [J Pn' * Jn];
    %     end
    %     H1 = J'*W1*J; % Hessian for missing data
    %     H1 = Psum*Pp' *H1 * Pp*Psum';
    %     norm(H-H1,'fro')
end

%% INVERSE Hessian
% Hi=inv(H+1e-8*eye(size(H)));
% Pa1=eye(Ia)-A(:,1)*A(:,1)'/sum(A(:,1).^2);
%crib1=sum(diag(Pa1*Hi(1:Ia,1:Ia)))/sum(A(:,1).^2) %% an approximate crib, to check correctness
len=Isum*r;
Ind=1:len;
for n = 1:N-1
    Ind(cI(n+1)+2-n:Isum+1-n:len-(n-1)*r)=[];
end
Hi2=inv(H(Ind,Ind)); % CRB(\theta)
%crib=sum(diag(Pa1*Hi2(1:Ia,1:Ia)))/sum(A(:,1).^2);
criball=zeros(1,r);
n = 1;
for k=1:r
    Pa1=eye(I(n))-U{n}(:,k)*U{n}(:,k)'/sum(U{n}(:,k).^2);
    k1=(k-1)*(Isum-N+1);
    criball(1,k)=sum(diag(Pa1*Hi2(k1+1:k1+I(n),k1+1:k1+I(n))))/sum(U{n}(:,k).^2);
    crbf(1,k)=trace(Hi2(k1+1:k1+I(n),k1+1:k1+I(n)));
end
