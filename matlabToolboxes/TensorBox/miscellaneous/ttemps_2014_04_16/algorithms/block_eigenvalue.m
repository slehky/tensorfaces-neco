function [X, C, evalue, residuums, micro_res, objective, elapsed_time] = block_eigenvalue(A, p, rr, opts)
%BLOCK_EIGENVALUE Calculate p smallest eigenvalues of a TTeMPS operator
%
%   [X, C, evalue, residuums, micro_res, objective, elapsed_time] = block_eigenvalue(A, P, RR, OPTS)
%       performs a block-eigenvalue optimization scheme to compute the p smallest eigenvalues of A
%       using the algorithm described in [1].
%
%   RR defines the starting rank and should usually be set to ones(1,d) where d is the dimensionality.
%   If p == 1, the algorithm equals a standard ALS-procedure for the eigenvalue problem, which is NOT
%   rank adaptive. Hence, in this case RR should be taken to be the expected rank of the solution or,
%   if unknown, the highest affordable rank.
%
%   Specify the wanted options using the OPTS argument. All options have
%   default values (denoted in brackets). Hence, you only have to specify those you want
%   to change.
%   The OPTS struct is organized as follows:
%       OPTS.maxiter        Maximum number of full sweeps to perform        [3]
%       OPTS.maxrank        Maximum rank during the iteration               [40]
%       OPTS.tol            Tolerance for the shift from one core
%                           to the next                                     [1e-8]
%       OPTS.tolLOBPCG      Tolerance for inner LOBPCG solver               [1e-6]
%       OPTS.maxiterLOBPCG  Max. number of iterations for LOBPCG            [2000]
%       OPTS.verbose        Show iteration information (true/false)         [true]
%       OPTS.precInner      Precondition the LOBPCG (STRONGLY RECOMMENDED!) [true]
%
%
%   NOTE: To run block_eigenvalue, Knyazev's implementation of LOBPCG is required. The corresponding
%         m-file can be downloaded from
%               http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
%
%	References:
%	[1] S.V. Dolgov, B.N. Khoromskij, I.V. Oseledets, D.V. Savostyanov
%		Computation of extreme eigenvalues in higher dimensions using block tensor train format
%	    Computer Physics Communications 185 (2014) 1207-1216.

%   TTeMPS Toolbox.
%   Michael Steinlechner, 2013-2014
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt


nn = A.size_col;
d = A.order;

if ~isfield( opts, 'maxiter');       opts.maxiter = 3;           end
if ~isfield( opts, 'maxrank');       opts.maxrank = 40;          end
if ~isfield( opts, 'tol');           opts.tol = 1e-8;            end
if ~isfield( opts, 'tolLOBPCG');     opts.tolLOBPCG = 1e-6;      end
if ~isfield( opts, 'maxiterLOBPCG'); opts.maxiterLOBPCG = 2000;  end
if ~isfield( opts, 'verbose');       opts.verbose = 1;           end
if ~isfield( opts, 'precInner');     opts.precInner = true;      end

if p == 1
    tolLOBPCGmod = opts.tolLOBPCG;
else
    tolLOBPCGmod = 0.01;
end

X = TTeMPS_rand( rr, nn );
% Left-orthogonalize the tensor:
X = orthogonalize( X, X.order );

% C is core of the current process node of size R_n x I x p x Rn+1
% This core extends TT-tensor (vector) to be a TT-matrix.
% The other cores of X are order-3 tensors of size Rk x I x Rk+1.

C = cell( 1, p );
C{1} = X.U{d};
for i = 2:p
    C{i} = rand(size(X.U{d}));
end

X.U{d} = rand( X.rank(d), X.size(d), X.rank(d+1), p );

evalue = [];
residuums = zeros(p,2*opts.maxiter);
micro_res = [];
resi_norm = zeros(p,1);
objective = [];
tic;
elapsed_time = [];


for i = 1:opts.maxiter
    
    disp(sprintf('Iteration %i:', i));
    % right-to-left sweep
    fprintf(1, 'RIGHT-TO-LEFT SWEEP. ---------------\n')
    
    for mu = d:-1:2
        sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];
        
        if opts.verbose
            fprintf(1,'Current core: %i. Current iterate (first eigenvalue): \n', mu);
            disp(X);
            fprintf(1,'Running LOBPCG: system size %i, tol %g ... ', prod(sz), max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod )))
        else
            fprintf(1,'%i  ',mu);
        end
        
        [left, right,Ax] = Afun_prepare( A, X, mu );
        
        try
            if opts.precInner
                expB = constr_precond_inner( A, X, mu );
                [V,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), [], ...
                    @(y) apply_local_precond( A, y, sz, expB), ...
                    max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod )), ...
                    opts.maxiterLOBPCG, 0);
            else
                [V,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                    opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
        catch
            [V,L] = eigs( @(y) Afun_block_optim( A, y, sz, left, right, mu),prod(sz),p,'SR');
            failureFlag = [];Lhist = [];L = diag(L);
        end
        
        if opts.verbose
            if failureFlag
                fprintf(1,'NOT CONVERGED within %i steps!\n', opts.maxiterLOBPCG)
            else
                fprintf(1,'converged after %i steps!\n', size(Lhist,2));
            end
            disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);
        end
        evalue = [evalue, L(1:p)];
        objective = [objective, sum(evalue(:,end))];
        elapsed_time = [elapsed_time, toc];
        
        X.U{mu} = reshape( V, [sz, p] );
        lamX = X;
        lamX.U{mu} = bsxfun(@times,lamX.U{mu},reshape(L,[1 1 1 p]));
        Ax.U{mu} = apply(A,X.U{mu},mu);res = Ax - lamX;
        res = orthogonalize(res,mu);
        resi_norm = sqrt(sum(reshape(res.U{mu},[],p).^2))';%norm(resi_norm2(:) - resi_norm(:))
        residuums(:,2*(i-1)+1)= resi_norm;
        
        
        %         lamX.U{mu} = repmat( reshape(L, [1 1 1 p]), [X.rank(mu), X.size(mu), X.rank(mu+1), 1]).*lamX.U{mu};
        %         res_new = apply(A, X) - lamX;
        %         for j = 1:p
        %             X.U{mu} = reshape( V(:,j), sz );
        %             Ax.U{mu} = apply(A,X.U{mu},mu);
        %             res = Ax - L(j)*X;
        %             resi_norm(j) = norm(res);
        %             residuums(j,2*(i-1)+1) = resi_norm(j);
        %         end
        micro_res = [micro_res, resi_norm];
        
        % split new core
        V = reshape( V, [sz, p] );
        V = permute( V, [1, 4, 2, 3] );
        V = reshape( V, [sz(1)*p, sz(2)*sz(3)] );
        
        [U,S,V] = svd( V, 'econ' );
        if p == 1
            s = length(diag(S));
        else
            s = find( diag(S) > opts.tol, 1, 'last' );
            s = min( s, opts.maxrank );
        end
        V = V(:,1:s)';
        X.U{mu} = reshape( V, [s, sz(2), sz(3)] );
        
        W = U(:,1:s)*S(1:s,1:s);
        W = reshape( W, [sz(1), p, s]);
        W = permute( W, [1, 3, 2]);
        
        % Move the expanding core to the core (\mu-1), i.e. the core
        % (\mu-1) will be of order-4 
        % The current core -mu  is expressed as order-3.
        for k = 1:p
            C{k} = tensorprod( X.U{mu-1}, W(:,:,k)', 3);
        end
        
        X.U{mu-1} = C{1};
        
        if opts.verbose
            disp( ['Augmented system of size (', num2str( [sz(1)*p, sz(2)*sz(3)]), '). Cut-off tol: ', num2str(opts.tol) ])
            disp( sprintf( 'Number of SVs: %i. Truncated to: %i => %g %%', length(diag(S)), s, s/length(diag(S))*100))
            disp(' ')
        end
    end
    
    % calculate current residuum
    
    fprintf(1, '---------------    finshed sweep.    ---------------\n')
    disp(['Current residuum: ', num2str(residuums(:,2*(i-1)+1).')])
    disp(' ')
    disp(' ')
    fprintf(1, '--------------- LEFT-TO-RIGHT SWEEP. ---------------\n')
    % left-to-right sweep
    for mu = 1:d-1
        sz = [X.rank(mu), X.size(mu), X.rank(mu+1)];
        
        if opts.verbose
            fprintf(1,'Current core: %i. Current iterate (first eigenvalue): \n', mu);
            disp(X);
            fprintf(1,'Running LOBPCG: system size %i, tol %g ... ', prod(sz), max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+2).^2))/sqrt(p)*tolLOBPCGmod )))
        else
            fprintf(1,'%i  ',mu);
        end
        
        [left, right,Ax] = Afun_prepare( A, X, mu );
        
        try
            if opts.precInner
                expB = constr_precond_inner( A, X, mu );
                [U,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), [], ...
                    @(y) apply_local_precond( A, y, sz, expB), ...
                    max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+2).^2))/sqrt(p)*tolLOBPCGmod )), ...
                    opts.maxiterLOBPCG, 0);
            else
                [U,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                    opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
        catch 
            [U,L] = eigs( @(y) Afun_block_optim( A, y, sz, left, right, mu),prod(sz),p,'SR');
            failureFlag = [];Lhist = [];L = diag(L);
        end
        
        if opts.verbose
            if failureFlag
                fprintf(1,'NOT CONVERGED within %i steps!\n', opts.maxiterLOBPCG)
            else
                fprintf(1,'converged after %i steps!\n', size(Lhist,2));
            end
            disp(['Current Eigenvalue approx: ', num2str( L(1:p)')]);
        end
        evalue = [evalue, L(1:p)];
        objective = [objective, sum(evalue(:,end))];
        elapsed_time = [elapsed_time, toc];
        
        
        X.U{mu} = reshape( U, [sz, p] );
        lamX = X;
        lamX.U{mu} = bsxfun(@times,lamX.U{mu},reshape(L,[1 1 1 p]));
        Ax.U{mu} = apply(A,X.U{mu},mu);res = Ax - lamX;
        res = orthogonalize(res,mu);
        resi_norm = sqrt(sum(reshape(res.U{mu},[],p).^2))';%norm(resi_norm2(:) - resi_norm(:))
        residuums(:,2*(i-1)+2)= resi_norm;
        
%         for j = 1:p
%             X.U{mu} = reshape( U(:,j), sz );
%             Ax.U{mu} = apply(A,X.U{mu},mu);
%             res = Ax - L(j)*X;
%             resi_norm(j) = norm(res);
%             res_o = orthogonalize(res,mu);resi_norm2(j) = norm(res_o.U{mu}(:));
%             residuums(j,2*(i-1)+2) = resi_norm(j);
%         end
        micro_res = [micro_res, resi_norm];
        
        % split new core
        U = reshape( U, [sz, p] );
        U = permute( U, [1, 2, 4, 3] );
        U = reshape( U, [sz(1)*sz(2), p*sz(3)] );
        
        [U,S,V] = svd( U, 'econ' );
        if p ==1
            s = length(diag(S));
        else
            s = find( diag(S) > opts.tol, 1, 'last' );
            s = min( s, opts.maxrank );
        end
        U = U(:,1:s);
        X.U{mu} = reshape( U, [sz(1), sz(2), s] );
        W = S(1:s,1:s)*V(:,1:s)';
        W = reshape( W, [s, p, sz(3)]);
        W = permute( W, [1, 3, 2]);
        
        % Redundant computation here - only need to compute for k = 1
        for k = 1:p
            C{k} = tensorprod( X.U{mu+1}, W(:,:,k), 1);
        end
        
        X.U{mu+1} = C{1};
        
        if opts.verbose
            disp( ['Augmented system of size (', num2str( [sz(1)*p, sz(2)*sz(3)]), '). Cut-off tol: ', num2str(opts.tol) ])
            disp( sprintf( 'Number of SVs: %i. Truncated to: %i => %g %%', length(diag(S)), s, s/length(diag(S))*100))
            disp(' ')
        end
    end
    
    fprintf(1, '---------------    finshed sweep.    ---------------\n')
    disp(['Current residuum: ', num2str(residuums(:,2*(i-1)+2).')])
    disp(' ')
    disp(' ')
end

evs = zeros(p,1);
% Evaluate rayleigh quotient for the p TT/MPS tensors
for i=1:p
    evec = X;
    evec.U{d} = C{i};
    evs(i) = innerprod( evec, apply(A, evec));
end
evalue = [evalue, evs];

end

function [left, right,y] = Afun_prepare( A, x, idx )
% A block-TT. A.U: (I1...Id) x (J1...Jd)
% x TT-tensor  (I1...Id)
% idx: mode
% left = tensor contraction between Aleft = [[A.U_{k<idx}]] and x_left = [[x.U_{k<idx}]]
%       left = x_left^T A_left * x_left : [1 x p]
% right = tensor contraction between Aright= [[A.U_{k>idx}]] and x_right = [[x.U_{k>idx}]]
%       right = x_right^T A_right * x_right : [1 x p]
%
%     y = A.apply(x);
y = A.apply(x,setdiff(1:A.order,idx)); % x'*A % Phan Anh Huy, Sept 2014

if idx == 1
    right = innerprod( x, y, 'RL', idx+1 );
    left = [];
elseif idx == x.order
    left = innerprod( x, y, 'LR', idx-1 );
    right = [];
else
    left = innerprod( x, y, 'LR', idx-1 );
    right = innerprod( x, y, 'RL', idx+1 );
end
end

function res = Afun_block_optim( A, U, sz, left, right, mu )

p = size(U, 2);

V = reshape( U, [sz, p] );
V = A.apply( V, mu );

if mu == 1
    tmp = reshape( permute( V, [3 1 2 4] ), [size(V, 3), sz(1)*sz(2)*p]);
    tmp = right * tmp;
    tmp = reshape( tmp, [size(right, 1), sz(1), sz(2), p]);
    tmp = ipermute( tmp, [3 1 2 4] );
elseif mu == A.order
    tmp = reshape( V, [size(V,1), sz(2)*sz(3)*p]);
    tmp = left * tmp;
    tmp = reshape( tmp, [size(left, 1), sz(2), sz(3), p]);
else
    tmp = reshape( permute( V, [3 1 2 4] ), [size(V, 3), size(V, 1)*sz(2)*p]);
    tmp = right * tmp;
    tmp = reshape( tmp, [size(right, 1), size(V, 1), sz(2), p]);
    tmp = ipermute( tmp, [3 1 2 4] );
    
    tmp = reshape( tmp, [size(V, 1), sz(2)*sz(3)*p]);
    tmp = left * tmp;
    tmp = reshape( tmp, [size(left, 1), sz(2), sz(3), p]);
    
end

res = reshape( tmp, [prod(sz), p] );


end

function res = apply_local_precond( A, U, sz, expB)
% 
%
p = size(U, 2);

x = reshape( U, [sz, p] );
res = zeros( [sz, p] );

for i = 1:size( expB, 1)
    tmp = reshape( x, [sz(1), sz(2)*sz(3)*p] );
    tmp = reshape( expB{1,i}*tmp, [sz(1), sz(2), sz(3), p] );
    
    tmp = reshape( permute( tmp, [2 1 3 4] ), [sz(2), sz(1)*sz(3)*p] );
    tmp = ipermute( reshape( expB{2,i}*tmp, [sz(2), sz(1), sz(3), p] ), [2 1 3 4] );
    
    tmp = reshape( permute( tmp, [3 1 2 4] ), [sz(3), sz(1)*sz(2)*p] );
    tmp = ipermute( reshape( expB{3,i}*tmp, [sz(3), sz(1), sz(2), p] ), [3 1 2 4] );
    
    res = res + tmp;
end
res = reshape( res, [prod(sz), p] );

end

