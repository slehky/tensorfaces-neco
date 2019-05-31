function [s,u,status]=mqp(H,fp,A,b,vlb,vub,x0,neq,nomsg)
% QP  
%    min  1/2 * s'*H*s + fp'*s   subject to   A*s + b = 0
%
%options=optimset('quadprog');
%[s,f,exitflag,output,LAMBDA]=quadprog(H,fp,A(1,:),b(1),[],[],[],[],x0,options);
% [H   -A]   [s ]   [-fb]
% [-A'  0] * [-u] = [-b ]
%  
vv = H\[fp A(1,:)'];

bHb = A(1,:)*vv(:,2);
lda =  -1/bHb  * (b + vv(:,2)'*fp);
s = -vv(:,1) -  vv(:,2) * lda;
% norm(dx-s)
% lda-LAMBDA.ineqlin
u = [lda;zeros(numel(s)*2,1)];
status = 'ok';
end