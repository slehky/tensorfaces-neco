function [U,D] = gen_matrix(I,R,c,pdtype)
%% This code generates (I x R) matrix U such that correlation coefficients
% c_rs = U(:,r)' * U(:,s)
% are equal to c or in range of [c1, c2]
%
% Phan Anh Huy, 23/08/2012
if nargin<4
    pdtype = 1;
end

c = unique(c);
if numel(c) == 1
    %C = c + (1-c) * eye(R);
    cr = zeros(R-1,1);br = zeros(R-1,1);
    cr(1) = c; br(1) = sqrt(1-cr(1).^2);
    for r = 2:R-1
        ss = sum(cr(1:r-1).^2);
        cr(r) = (c - ss)/br(r-1);
        br(r) = sqrt(1-ss - cr(r).^2);
    end
    
    % Generate diagonal matrix
    D = diag([1;br]);
    for r = 1:R-1
        D(r,r+1:end) = cr(r);
    end
else %  offdiag(C = U'*U) in [c1,c2]
    c = c(1:2);
    % Generate a random correlation matrix with coeffs in a range of [-1 1]
    C = RandomCorr(R,1);
     
    if pdtype == 1
        % Transform C to a new correlation matrix with coefficients in [c(1),c(2)]
        % x * C + y + (1-x-y) *eye(R)  where x = (c2-c1)/2, y = (c2+c1)2
        C = (c(2)-c(1))/2 * C + (c(2)+c(1))/2 + (1-c(2)) * eye(R);
    elseif pdtype == 2
        C = abs(C);C2 = C;C2(1:R+1:end) = [];x = [min(C2(:)) max(C2(:))];
        alpha = (c(2)-c(1))/(x(2)-x(1));
        beta = c(2) - alpha*x(2);
        C = alpha * C + beta + (1-alpha - beta) * eye(R);
    end

    [U,S] = eig(C);
    D = U * diag(sqrt(diag(S)));
    D = D';
end

V = orth(randn(I,R));
U = V*D(1:R,1:R);
end



function [C,p]=RandomCorr(n,K)
%   RandomCorr is a function for generating random correlation matrix.
%   The only input (n) is a dimension, which must be more than two.
%   In this algorithm, the correlation coefficients are uniformly
%   distributed within its boundaries that are sequentially computed.
%
%   Reference: Numpacharoen, K. and Atsawarungruangkit, A.,
%   Generating random correlation matrices based on the boundaries of their coefficients
%   (August 24, 2012). Available at SSRN: http://ssrn.com/abstract=2127689
%
%   Kawee Numpacharoen and Amporn Atsawarungruangkit, August 112, 2012.
if length(n)>1,error('Dimension must be scalar!!'),end
if n<2,error('Dimension must be larger than two!'),end
if mod(n,1)~=0,error('Dimension must be integer!'),end
C=zeros(n);
B=tril(ones(n),0);
p=-1;
while p < 0
    %Step 1: calculate correlation coefficients in the first column
    C(2:n,1)=-1+2*round(rand(n-1,1)*10^8)/(10^8);
    B(2:n,1)=C(2:n,1);
    for i= 2:n
        B(i,2:i)=sqrt(1-C(i,1)^2);
    end
    
    %Step 2: calculate the rest of correlation coefficients
    for i=3:n
        for j= 2:i-1
            B1=B(j,1:j-1)*B(i,1:j-1)';
            B2=B(j,j)*B(i,j);
            Z=B1+B2;
            Y=B1-B2;
            if B2 < K, C(i,j)=B1;cosinv=0; else
                C(i,j)=Y+(Z-Y)*round(rand(1)*10^8)/(10^8);
            end
            
            cosinv=(C(i,j)-B1)/B2;
            
            if isfinite(cosinv)==false
                
            elseif cosinv >1
                B(i,j)=B(i,j);
                B(i,j+1:n)=0;
            elseif cosinv <-1
                B(i,j)=-B(i,j);
                B(i,j+1:n)=0;
            else
                B(i,j)=B(i,j)*cosinv;
                sinTheta=sqrt(1-cosinv^2);
                for k=j+1:n
                    B(i,k)=B(i,k)*sinTheta;
                end
            end
        end
    end
    C=C+C'+eye(n);
    
    %Step 3: randomly reorder the correlation matrix.
    order = randperm(n);
    C=C(order,order);
    %Step 4: check validity and adjust to a valid one.
    p=min(eig(C));
end
end

% % Generate upper triangular matrix D of size R x R
% % D = [ 1  c1 c1 ... c1
% %          b1 c2 ... c2
% %             b2 ... c3
% %                ... :
% %                    cR
% %                    bR]
% %
% % D' * D = c * I_R - c 1_RxR
% %
% if strcmp(class(c),'sym')
%     cr = sym('cr',[R-1,1]);br = sym('br',[R-1,1]);
% else
%     cr = zeros(R-1,1);br = zeros(R-1,1);
% end
% 
% if numel(c) == 1
%     cr(1) = c; br(1) = sqrt(1-cr(1).^2);
%     for r = 2:R-1
%         ss = sum(cr(1:r-1).^2);
%         cr(r) = (c - ss)/br(r-1);
%         br(r) = sqrt(1-ss - cr(r).^2);
%     end
%     
%     % Generate diagonal matrix
%     D = diag([1;br]);
%     for r = 1:R-1
%         D(r,r+1:end) = cr(r);
%     end
%     
%     % Generate factors
%     U = orth(randn(I,R));
%     U = U * D(1:R,1:R);
%     
% else % Generate U such that offdiag(C = U'*U) in range [c1,c2]
%     % 20/02/2012
%     c = sort(c);c = c(1:2); %
%     C = reshape(1:R^2,R,R);
%     ind = find(triu(C,1));
%     
%     gendone = false;
%     %figure(1); clf; hold on
%     for it2 = 1:10
%         ca = randn(1,R*(R-1)/2);
%         for iter = 1:10
%             ca = (ca-min(ca))/(max(ca)-min(ca));
%             ca = ca * (c(2) - c(1)) + c(1);
%             C = eye(R);
%             C(ind) = ca;
%             C = max(C,C');%Ca0 = C;
%             [u,s] = eig(C);
%             
%             s = diag(s);
%             if any(s<=0)
%                 %s'
%                  %plot(s); hold on
% %                 s = s - 2*min(s);
% %                 idneg = find(s<0);
% %                 [ssort,idsort] = sort(s,'descend');
% %                 s(idsort(1:numel(idneg))) = ssort(1:numel(idneg)) + 2*sort(s(idneg),'ascend');
%                 s = abs(s);
%                 %[smax,idmax] = max(s);
%                 %[smin,idmin] = min(s);
%                 %idset = setdiff(1:R,[idmin idmax]);
%                 %s(idset) = s(idset)*(R-smax-smin)/(sum(s) - smax - smin);
%                 s = (s/sum(s) * R);
%                 C2 = u*diag(s) * u';
%                 diagc = sqrt(diag(C2));
%                 C2 = diag(1./diagc) * C2 * diag(1./diagc);
%                 ca = C2(ind);
%                 ca = (ca-min(ca))/(max(ca)-min(ca));
%                 ca = ca * (c(2) - c(1)) + c(1);
%             else
%                 gendone = true;
%                 break
%             end
%         end
%         if gendone
%             break
%         end
%     end
%         
% 
% %     % 20/02/2012
% %     c = sort(c);c = c(1:2); %
% %     C = reshape(1:R^2,R,R);
% %     ind = find(triu(C,1));
% %     ca = randn(1,R*(R-1)/2);
% %  
% %     ca = (ca-min(ca))/(max(ca)-min(ca));
% %     ca = ca * (c(2) - c(1)) + c(1);
% %     C = eye(R);
% %     C(ind) = ca;
% %     C = max(C,C');%Ca0 = C;
% %     [u,s] = eig(C);
% %     
% %     s = abs(diag(s));
% %   
%     s = (s/sum(s) * R);
%     D = u * diag(real(sqrt(s)));
%     D = D';
%     
%     C = abs(D'*D);
%     C = triu(C,1);
%     C = C(ind);
%     
%     
%     if (c(1) >  (min(C(:)) +1e-6)) || (c(2) < (max(C(:)) - 1e-6)) %|| ~gendone
%         warning('Correlation degrees may not be in the expected range.')
%     end
%     
%     if I< R
%         [V,foe] = qr(randn(I,R)',0);
%         V = V';
%     else
%         V = orth(randn(I,R));
%     end
%     U = V * D;
%     %Ca = U'*U;
% end
% end