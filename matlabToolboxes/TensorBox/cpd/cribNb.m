function [crb B0]=cribNb(C,Ia)
%
% computing crib for 2 rank tensors of arbitrary dimension, with
% correlations of factors given in vector c; Ia is the length of the first
% factor matrix (number of rows).
%
% This procedure can be run symbolically. For example,
% declare "syms a b c Ia" and call "crib2([a b c],Ia)" to get symbolic result.
%
[r r1 N]=size(C);
%I=find(abs(C)<1e-5);
%C(I)=1e-5*sign(C(I));
G=C; H=C;
X=C;
for n=1:N
    X(:,:,n)=C(:,:,n)-C(:,1,n)*C(:,1,n).'/C(1,1,n);
    %G(:,:,n)=prod(C(:,:,[1:n-1 n+1:N]),3);
    %H(:,:,n)=prod(C(:,:,[2:n-1 n+1:N]),3); %% Gamma_{1n}
    G(:,:,n)=prodm(C(:,:,[1:n-1 n+1:N]));
    H(:,:,n)=prodm(C(:,:,[2:n-1 n+1:N])); %% Gamma_{1n}
end
G1=inv(G(:,:,1));
%dG1=diag(reshape(G(:,:,1),1,r^2));
g1=G1(1,:);
g11=G1(1,1);
in2=reshape(reshape(1:r^2,r,r)',r^2,1);
P=eye(r^2);
P=P(in2,:);
W=zeros(r^2,r^2); Y=W; 
GC1=kron(inv(G(:,:,1)),C(:,:,1));
for n=2:N
    K1=P*diag(reshape(H(:,:,n),1,r^2));
    K4=diag(reshape(G(:,:,n)./C(:,:,n),1,r^2))*P;
    Psi2=kron(inv(G(:,:,n)),X(:,:,n));
    Z0=eye(r^2)-Psi2*K4;
 %   iZ0=inv(Z0);
 %   Psi2t=kron(inv(G(:,:,n)),X(2:r,2:r,n));
 %   E=kron(eye(r),[zeros(r-1,1) eye(r-1)]);
 %   iZ0=eye(r^2)+E'/(eye(r^2-r)-Psi2t*E*K4*E')*Psi2t*E*K4;
 %   aux=E*K4*E'; F=aux(r:r^2-r,r:r^2-r);
 %   FPi=eye(r^2-2*r+1)-F*Psi2t(r:r^2-r,r:r^2-r)
 %   iPF=eye(r^2-r)+Psi2t(:,r:r^2-r)/FPi*aux(r:r^2-r,:);
 %   iZ0=eye(r^2)+E'*iPF*Psi2t*E*K4;
    %   K5=Psi2*();
    %Z0
    %det(Z0)
    K1P=K1/Z0*Psi2;
    W=W+K1P*diag(reshape(C(:,:,1)./C(:,:,n),1,r^2));
    Y=Y+K1P*K1;
 %   W2=W2+K1P*diag(reshape(G(:,:,n),1,r^2));
end  
V=W-Y*GC1;
B0=(V/(eye(r^2)+V)-eye(r^2))*Y;
crb=(Ia-1)*g11-sum(diag(B0*kron(g1.'*g1,X(:,:,1))));
crb=crb/C(1,1,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X=prodm(C)
[r r1 N]=size(C);
X=squeeze(C(:,:,1));
for k=2:N
    X=X.*C(:,:,k);
end    
end