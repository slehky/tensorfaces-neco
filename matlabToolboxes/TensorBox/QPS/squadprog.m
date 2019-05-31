function [x,fval,lda] = squadprog(Q,b,solving_method,verbosity)
% Quadratic programing with spherical constraints
%    min f(x) = 1/2*x'*Q*x + b'*x
%    subject to  x'*x = 1
%
% where Q is a positive semi-definite matrix of size K x K
% b is a Kx1 vector.
%
% Q can be given in form of eigenvale decomposition
% Q.U Q.s where eigenvalues are sorted in ascending order.
%
% PHAN ANH-HUY, July 2015.

% Objective function
% fcost = @(x) 1/2*x'*Q*x + b'*x;


if nargin<3
    solving_method = 'fzero';
end
if solving_method == 1
    solving_method = 'fminbnd';
end

if nargin<4
    verbosity = 0;
end

bn = norm(b);

if bn == 0
    % Compute EVD of Q
    [U,s] = eig(Q);
    [s,id] = min(diag(s));
    x = U(:,id);
    fval = s;
    lda = 1;
    return
end

if isnumeric(Q)
    K = size(Q,1);
    
    % Compute EVD of Q
    if issparse(Q)
        [U,s] = eigs(Q,size(Q,1));
    else
        [U,s] = eig(Q);
    end
    [s,id] = sort(diag(s),'ascend');
    U = U(:,id);
    
elseif isstruct(Q) && all(isfield(Q,{'U','s'}))
    try
        U = Q.U;
        s = Q.s;
        K = size(U,1);
    catch
        return;
    end
end

s1 = s(1);s0 = s;

% Normalize Q, b and c
c = U'*b;
%bn = b/bn;
c = c/bn;

% Scale and shift the matrix Q so that it becomes positive definite
% with the smallest eigevalue of 1, i.e. sigma_1 = 1.
%
%Qn = Q/bn+(1-s(1)/bn)*speye(I);
s = s/bn+1-s(1)/bn;

% The ell-2 unit-length vector x is expressed as
%       x = (Q - lda * I)\b
%  where lda is a smallest root of the rational function in interval [0,1]
%     sum((c./(lda-s)).^2) == 1
%
% Construct an equipvalent polynomial of degree 2K



%% If c(1) == 0, x(1) = 0,
% solve the reduced problem for x(2:K), s(2:K)
% round c, small c will treat as zeros.
c(abs(c)<1e-12) = 0;

zero_cid = c == 0;

if any(zero_cid)
    %
    nzero_cid = ~zero_cid;
    s2 = s(nzero_cid);
    c2 = c(nzero_cid);
    
    if c(1) == 0 
         
        x = zeros(K,1);
        x(nzero_cid) = c2./(s(1)-s2);
        normx2 = norm(x(nzero_cid));
        
        if normx2 < 1
            % there are two vectors which are different only in x(1)
            % x(1) = sqrt(1-normx2^2)  or -sqrt(1-normx2^2);
            warning('There may be two different solutions.')
            x(1) = sqrt(1-normx2^2);
            x = [x  x];
            x(1) = -x(1);
            
            lda = 1;
            fval = 1/2*s'*(x.^2) + c'*x;
            
        else
            % Solve the reduced problem with x(1)= 0;
            % shift s so that s(1) = 1;
            
            if numel(s2)>1
                s2 = s2-s2(1)+1;
                [xreduced,fval,lda] = solve_diagonal_qp(s2,c2,solving_method,verbosity);
                
                x(1) = 0;
                nzero_cid(1) = false;
                x(nzero_cid) = xreduced;
                
                lda = lda + s2(2)-1;
                fval = 1/2*(x.^2)'*s + c'*x;
                
            else % numel(s2) == 1
                lda = s2-c2;
                x(1) = 0;
                nzero_cid(1) = false;
                x(nzero_cid) = -1;
                
                fval = 1/2*s + c;
            end
        end
        
    else
 
        [x2,fval,lda] = solve_diagonal_qp(s2,c2,solving_method,verbosity);
        x = zeros(K,1);
        x(nzero_cid) = x2;
    end
    
else
    [x,fval,lda] = solve_diagonal_qp(s,c,solving_method,verbosity);
end

%% Compute the solution x
fval = (1/2*(s'*x.^2) + c'*x)*bn+(s0(1)-bn)/2;

% x  = -(Qn-lda*eye(I))\bn;
x = U*x;

% Compute the objective value
% fval = ((lda +sum(c.^2./(lda-s)))*bn + s1 - bn)/2;



end


%%
function [x,fval,lda] = solve_diagonal_qp(s,c,solving_method,verbosity)
%
%   minimize  1/2*x'*diag(s)*x + c'*x
%   subject to   x'*x = 1;
%
%   where sK>...>s2>s1 = 1
%   and    c'*c = 1;
%          c(k) ~= 0 for all k.
%
%
%   Solution x = [ck/(lda - s(k)]
%       where
%            lda is the minimum root in [0, 1-c(1)] of the function
%
%           f(lda) = 1 - sum_k  c_k^2 /(lda - s_k)^2
%
%  solving_method:  
%     poly_root: solution as root of poynomial
%     fzero    : find lda as root of 1-st derivative 
%     fminbnd  : find lda as minimiser to the function 
%                   f(lda) = -1/2*(x+sum(c^2/(x-s))
%
if nargin<3
    solving_method = 'fzero';
end
if nargin<4
    verbosity = 0;
end

K = numel(c);

if K == 1
    lda = s-c;
else
    
    if (strcmp(solving_method,'polyroot') && (K<10))
        pp = 0;
        for n = 1:K
            sr = s([1:n-1 n+1:end]); sr = [sr ; sr];
            pp = pp +poly(sr)*c(n)^2;
        end
        
        pp = poly([s ;s]) - [0 0 pp];
        % p(1) == 1
        
        if max(abs(pp)) < 1e8
            A = diag(ones(2*K-1,1),-1);
            A(1,:) = -pp(2:end);
            lda = eigs(A,1,'SM');
        else
            lda = roots(pp);
            lda = min(lda);
        end
    else
        % OR solve the rational function with solution in [1-|c1|*t1, 1-|c1|*t2]
        %
        %     g_lda = @(t) sum(c.^2./(t-s).^2) -1;
        %
        %     lda = fzero(g_lda, );
        
        % FInd a good initial point in (0, 1) for the following function
        %    f(lda) = sum_k  c_k^2/(lda - sigma_k)^2 -1
        % f(1-|c1|*t1) < 0, f(1+|c1|*t2)  > 0
        
        % Added on 24/06/2016
        % the new bound of lambda :   1-abs(c1)* t1 < lambda < 1 - abs(c1) * t2
        % where t1 and t2 >=1 and <= 1/abs(c1), are roots of polynomial of
        % degree-4
        if numel(s) >=2
            
            if ((s(2)-s(1))< 1)  && ((s(end)-s(1))>1) && (numel(s)>10)
                [lower_bound,upper_bound] = bound_lambda_truncated(c,s);
            else
                [lower_bound,upper_bound] = bound_lambda(c,s);
            end
            
            if isempty(lower_bound)
                lower_bound = 1e-50;
            end
            if isempty(upper_bound)
                upper_bound = 1-abs(c(1));
            end
        else
            lower_bound = 1e-50;
            upper_bound = 1-abs(c(1));
        end
        
        % f(lda) = sum c_k^2./(s_k-lda)^2 -1
        % This is the first derivative of the function
        %   f(lda) = sum c_k./(s_k-lambda) - lambda
        
        f0_lda = @(t) sum(bsxfun(@rdivide,c,bsxfun(@minus,s,t)).^2)-1;
        
        lda_i = logspace(log10(lower_bound),log10(upper_bound),20);
        lda_i = sort([lda_i 1-abs(c(1))]);
        
        f_i = f0_lda(lda_i);
        lda_i = [0 lda_i 1];
        f_i = [-inf f_i inf];
        id = find(f_i>0,1);
        
        lda_1 = lda_i(id-1);
        lda_2 = lda_i(id);
        
        % June 2, if |lda_2 - lda_1| is too small, lda is (lda1+lda2)/2 ,
        % no need to use fzero
        if abs(lda_2-lda_1)<=1e-15
            lda = (lda_1+lda_2)/2;
            gval = f0_lda(lda);
        else
            
            c2 = c.^2;
            
            % Solve by fzero
            % solving_method = 'fzero'; % fminbnd
            %solving_method= 'fminbnd';
            %solving_method= 'fsolve';
            
            switch solving_method
                case 'fzero'
                    % find zero of the 1st derivative
                    % f'(lda) = -(1 - sum(c^2./(lda - s).^2))
                    %options = optimset('fzero');
                    %[lda,gval] = fzero(@(x) f0_lda(x),[lda_1 lda_2],options);
                    [lda,gval] = fzero(@(x) f0_lda(x),[lda_1 lda_2]);
                    
                case 'fsolve'
                    
                    options = optimoptions('fsolve','Jacobian','on');
                    %options = optimoptions('fsolve','Jacobian','on','JacobMult',@(Jinfo,x,flag,g) jmfun(Jinfo,x,flag,f0_lda));
                    lda_0 = (lda_1+lda_2)/2;
                    [lda,gval] = fsolve(@(x) g_lda(x,c2,s),lda_0,options);
                    
                    
                case 'fminbnd'
                    
                    % Solve by fminbound   1-abs(c1)*t1 < lda < 1 - abs(c1)*t2
                    %          minimize  f(lda)  = -(lda + sum(c^2./(lda - s)))
                    %   1-st derivative  f'(lda) = -(1   - sum(c^2./(lda - s).^2))
                    %   2-st derivative f''(lda) = -(2*sum(c^2./(lda - s).^3))
                    % The objective function f(lda) is changed sign so that it is convex in
                    % the bound
                    
                    f1 = f_lda(lda_1,c2,s);
                    f2 = f_lda(lda_2,c2,s);
                    lda = (lda_1+lda_2)/2;
                    gval = f0_lda(lda);
                    
                    options = optimset('fminbnd');
                    options.TolX = 1e-15;
                    options.gradTolX = 1e-10;
                    
                    while (abs(f2-f1)>1e-30) || (abs(gval) > options.gradTolX)
                        lda = fminbnd(@(x) f_lda(x,c2,s),lda_1,lda_2,options);
                        %lda = fminbnd(@(x) f_lda2(x,c2,s),lda_1,lda_2,options);
                        
                        if (lda_2-lda) < options.TolX
                            break
                        end
                        % Check the gradient and refine the root if necessary
                        gval = f0_lda(lda);
                        if abs(gval) < options.TolX
                            break
                        else
                            if gval<0
                                lda_1 = lda;
                                f1 = f_lda(lda,c2,s);
                            else
                                lda_2 = lda;
                                f2 = f_lda(lda,c2,s);
                            end
                        end
                    end
            end
        end
        if verbosity == 1
            fprintf('Lda is %d, should be in the interval [%d, %d].\n',lda,lda_1,lda_2)
            fprintf('Gradient at solution is %d, which should be zero.\n',gval);
        end
    end
end

x = (c./(lda-s));
%fval = 1/2*(x.^2)'*s + c'*x; %
fval = (lda+c'*x)/2;%1/2*(lda + sum(c2./(lda-s)));

end


%%
function [f,g,H] = g_lda(x,c2,s)
% g(x) = - (sum (c)^2 ./(x-s)^2 - 1)
%
xs = x-s;
xs2 = xs.^2;
f = sum(c2./xs2) -1;

if nargout >1
    % 1st derivative 
    g = -2*sum(c2./(xs2.*xs));
end
if nargout >2
    % 2nd derivative 
    H = 6*sum(c2./xs2.*xs2);
end
end


function [f,g,H] = f_lda(x,c2,s)
% c2 = c.^2
% f(x) = -(x + sum_k (ck^2/(x-sk)))
%
% fminbnd uses only the fval, neither the 1st and 2nd derivatives
% 
xs = x-s;
f = -(x + sum(c2./xs));
%
if nargout >1
    xs2 = xs.^2;
    % 1-st derivative of f(x) w.r.t x
    g = -(1 - sum(c2./xs2));
end
if nargout >2
    % 2-nd derivative of f(x) w.r.t x
    % alway positive in the search interval
    H = -2*sum(c2./xs2.*xs);
end
end


%%
function [lower_bound,upper_bound] = bound_lambda(c,s)
%
%
%% compute lower-bound of lambda
c1 = abs(c(1));

if 1>=1/abs(c(1))
    lower_bound = 0;
    upper_bound = 0;
    return;
end

d = s(2)-s(1);
p_4 = [c1^2 2*c1*d d^2-1 -2*c1*d -d^2];

t1 = roots(p_4);
t1 = t1(abs(imag(t1))<1e-8);
idt =  abs(1-t1)<1e-5;
t1(idt) = 1;
    
% t1 = round(real(t1)*1e8)/1e8;
t1 = t1((t1>=1) & (t1<= 1/c1));
t1 = min(t1);

lower_bound = 1-c1*t1;

%% compute upper-bound of lambda
d = s(end)-s(1);
p_4 = [c1^2 2*c1*d d^2-1 -2*c1*d -d^2];

t2 = roots(p_4);
t2 = t2(abs(imag(t2))<1e-8);
idt =  abs(1-t2)<1e-5;
t2(idt) = 1;
% t2 = round(real(t2)*1e8)/1e8;
t2 = t2((t2>=1) & (t2<= 1/c1));
t2 = max(t2);

upper_bound = 1-c1*t2;

end

%%
function [lower_bound,upper_bound] = bound_lambda_truncated(c,s,L)
% FInd bound of lambda of 
% f(lda) = sum_k  c_k^2/(lda - sigma_k)^2 -1
% 
%  by solving truncated equation 
%
%   f(lda) = sum_{k=1}^{L}  c_k^2/(lda - sigma_k)^2 -1

%% compute lower-bound of lambda
% 
% Find the last s(k) which exceeds 1

if nargin < 3
    L = find(s>2,1,'first');
    L = max(2,min(20,min(round(numel(s)/10),L)));
end
solving_method = 'fzero'; 
verbosity = 0;

% Lower bound 
cL = [c(1:L); sqrt(sum(c(L+1:end).^2))];
sL = [s(1:L+1)];
[x_lw,fval_lw,lower_bound] = solve_diagonal_qp(sL,cL,solving_method,verbosity);

% Upper bound 
sL = [s(1:L) ; s(end)];
[x_up,fval_up,upper_bound] = solve_diagonal_qp(sL,cL,solving_method,verbosity);
 
end




% %%
% function W = jmfun(Jinfo,Y,flag,g)
% % g: function handle computes the 1-st derivative 
% %    g = @(x) sum(bsxfun(@rdivide,c,bsxfun(@minus,s,x)).^2)-1;
% %
% % If flag == 0, W = J'*(J*Y).
% % If flag > 0, W = J*Y.
% % If flag < 0, W = J'*Y.
%  
% if flag == 0
%      W = g(g(Y));
% elseif flag > 0
%     W = g(Y);
% else
%     W = g(Y);
% end
% end