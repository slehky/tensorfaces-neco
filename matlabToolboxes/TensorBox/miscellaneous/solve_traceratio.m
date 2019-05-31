
function [u,fval] = solve_traceratio(Sb,Sw,ncomps,method,u0)
% max trace(u'*Sb*u)/trace(u'*Sw*u)

%% Fast computing
Sb = max(Sb,Sb');
Sw = max(Sw,Sw');

switch method
    case 'gevd'
        %                 opts.disp = 0;
        %                 opts.cholB = true;
        %                 [u,d] = eigs(Sb,chol(Sw),Ps(n),'LM',opts);
        
        [u,d] = eig(Sb,Sw,'chol');
        %                 [u,d] = eig(Sw\Sb);
        d = diag(d);
        [d,id] = sort(d,'descend');
        u =  u(:,id(1:ncomps));
        fval = sum(d(1:ncomps));
        
    case 'traceratio'  
        eigsopts.disp = 0;
        if isempty(u0)
            [u,lambda] = solve_traceratio(Sb,Sw,ncomps,'gevd');
        else
            u = u0;
            [u,ru] = qr(u,0);
        end
        fval = trace(u'*Sb*u)/trace(u'*Sw*u);
        lambda_ = fval;
        while 1
            %if ncomps == size(Sb,1)
            [u,d] = eig(Sb - fval*Sw);
            d = diag(d);
            d(abs(d)<1e-10) = 0;
            [d,id] = sort(d,'descend');
            id2 = find(cumsum(d(1:ncomps))>=0,1,'last');
            u = u(:,id(1:id2)); 
            %ncomps = id2;
            %else
            %[u,d] = eigs(Sb - lambda*Sw,ncomps,'LM',eigsopts);
            %end
            %u = u*u';

            Sw2 = u'*Sw*u;
            Sb2 = u'*Sb*u;
            [u2,d] = eig(Sb2,Sw2,'chol');
             u = u*u2;
%             
%             S = (u'*Sw*u);
%             S = max(S,S');
%             %         if ncomps == size(Sb,1)
%             [u2,d] = eig(S);
%             u = u*u2;
            %         else
            %             [u,d] = eigs(S,ncomps,'LM',eigsopts);
            %         end
            fval = trace(u'*Sb*u)/trace(u'*Sw*u);
            
            
            if abs(fval-lambda_(end))< 1e-5*lambda_(end)
                break
            end
            lambda_ =[lambda_ fval];
        end
%         lambda_
end
end