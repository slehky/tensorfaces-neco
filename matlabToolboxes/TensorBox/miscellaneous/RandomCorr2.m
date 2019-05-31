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
    %C(2:n,1)=-1+2*round(rand(n-1,1)*10^8)/(10^8);
    C(2:n,1) = rand(n-1,1) *(K(2)-K(1)) + K(1);
        
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
            
            Y = max(K(1),Y);
            Z = min(Z,K(2));
            if Z < Y,
                C(i,j)= K(1); cosinv=0; 
            elseif Y>K(2)
                C(i,j) = K(2);
            else
                C(i,j)=Y+(Z-Y)*round(rand(1)*10^8)/(10^8);
            end
            if ((C(i,j)<K(1)) || (C(i,j) > K(2)))
                1
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

