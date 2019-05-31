function [X, C, evalue, residuums, micro_res, objective, elapsed_time] = amen_eigenvalue(A, prec, p, rr, opts)
    %AMEN_EIGENVALUE Calculate p smallest eigenvalues of a TTeMPS operator
    %
    %   [X, C, evalue, residuums, micro_res, objective, elapsed_time] = amen_eigenvalue(A, PREC, P, RR, OPTS)
    %       performs a rank-adaptive AMEn-type optimization scheme to compute the p smallest eigenvalues of A
    %       using the algorithm described in [1].
    %
    %   A preconditioner can be given by passing the PREC argument:
    %       []      no preconditioner used
    %       PR      with PR of type TTeMPS_op: PR is applied as a preconditioner
    %       true    exponential sum preconditioner described in [1] (recommended setting)    
    %
    %   RR defines the starting rank and should usually be set to ones(1,d) where d is the dimensionality.
    %   
    %   Specify the wanted options using the OPTS argument. All options have
    %   default values (denoted in brackets). Hence, you only have to specify those you want 
    %   to change.
    %   The OPTS struct is organized as follows: 
	%       OPTS.maxiter        Maximum number of full sweeps to perform        [3]
	%       OPTS.maxrank        Maximum rank during the iteration               [40]
	%       OPTS.maxrankRes     Maximum rank of residual (0 = No limit)         [0]
	%       OPTS.tol            Tolerance for the shift from one core 
    %                           to the next                                     [1e-8]
	%       OPTS.tolOP          Tolerance for truncation of residual            [1e-6]
	%       OPTS.tolLOBPCG      Tolerance for inner LOBPCG solver               [1e-6]
	%       OPTS.maxiterLOBPCG  Max. number of iterations for LOBPCG            [2000]
	%       OPTS.verbose        Show iteration information (true/false)         [true]
	%       OPTS.precInner      Precondition the LOBPCG (STRONGLY RECOMMENDED!) [true]
    %
    %
    %   NOTE: To run amen_eigenvalue, Knyazev's implementation of LOBPCG is required. The corresponding
    %         m-file can be downloaded from 
    %               http://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
    %
	%	References: 
	%	[1] D. Kressner, M. Steinlechner, A. Uschmajew:
	%		Low-rank tensor methods with subspace correction for symmetric eigenvalue problems
	%		MATHICSE Technical Report 40.2013, December 2013. Submitted.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


nn = A.size_col;
d = A.order;

if ~isempty(prec)
    if isa(prec,'TTeMPS_op')
        precondition = 1;
    else
        precondition = 2;
    end
else
    precondition = 0;
end

if ~isfield( opts, 'maxiter');       opts.maxiter = 3;           end
if ~isfield( opts, 'maxrank');       opts.maxrank = 40;          end
if ~isfield( opts, 'maxrankRes');    opts.maxrankRes = 0;        end
if ~isfield( opts, 'tol');           opts.tol = 1e-8;            end
if ~isfield( opts, 'tolOP');         opts.tolOP = 1e-6;          end
if ~isfield( opts, 'tolLOBPCG');     opts.tolLOBPCG = 1e-6;      end
if ~isfield( opts, 'maxiterLOBPCG'); opts.maxiterLOBPCG = 2000;  end
if ~isfield( opts, 'verbose');       opts.verbose = true;        end
if ~isfield( opts, 'precInner');     opts.precInner = true;     end

if p == 1
    tolLOBPCGmod = opts.tolLOBPCG;
else
    tolLOBPCGmod = 0.01;
end

X = TTeMPS_rand( rr, nn ); 
% Left-orthogonalize the tensor:
X = orthogonalize( X, X.order );

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
            fprintf(1,'Running LOBPCG: system size %i, tol %g ... ', prod(sz),  max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod ))) %opts.tolLOBPCG)
        else
            fprintf(1,'%i  ',mu);
        end
       
        [left, right,Ax] = Afun_prepare( A, X, mu );
        
        try
            if opts.precInner
                expB = constr_precond_inner( A, X, mu );
                [V,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), [], ...
                    @(y) apply_local_precond( y, sz, expB), ...
                    max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+1).^2))/sqrt(p)*tolLOBPCGmod )), opts.maxiterLOBPCG, 0);
            else
                [V,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                    opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
        catch
            [V,L] = eigs( @(y) Afun_block_optim( A, y, sz, left, right, mu),prod(sz),p,'SM');
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
        %lamX.U{mu} = repmat( reshape(L, [1 1 1 p]), [X.rank(mu), X.size(mu), X.rank(mu+1), 1]).*lamX.U{mu};
        lamX.U{mu} = bsxfun(@times,lamX.U{mu},reshape(L,[1 1 1 p]));
        
        %res = apply(A, X) - lamX;
        Ax.U{mu} = apply(A,X.U{mu},mu);res = Ax - lamX;
        
        if opts.maxrankRes ~= 0
            res_sz = [size(res.U{mu},1), size(res.U{mu},2), size(res.U{mu},3), size(res.U{mu},4)];
            res.U{mu} = reshape( permute( res.U{mu}, [1 2 4 3]), [ res_sz(1), res_sz(2)*p, res_sz(3)]);
            res = truncate( res, [1 opts.maxrankRes*ones(1,d-1) 1] );
            res.U{mu} = ipermute( reshape( res.U{mu}, [res.rank(mu), res_sz(2), p, res.rank(mu+1)] ), [1 2 4 3]);
        end

        % A not so nice way to calculate the residual norm
%         for j = 1:p
%             tmp = res;
%             tmp.U{mu} = tmp.U{mu}(:,:,:,j);
%             resi_norm(j) = norm(tmp);
%             residuums(j,2*(i-1)+1) = resi_norm(j);
%         end
%         % Fast computation of the residual norm
        res_o = orthogonalize(res,mu); 
        resi_norm = sqrt(sum(reshape(res_o.U{mu},[],p).^2))';%norm(resi_norm2(:) - resi_norm(:))
        residuums(:,2*(i-1)+1)= resi_norm;
        micro_res = [micro_res, resi_norm];
        
        if precondition == 1
            res = apply(prec, res);
        end
        
        res = contract( X, res, [mu-1, mu]);
        res_combined = unfold(res{1},'left') * reshape(res{2}, [size(res{2},1), size(res{2},2)*size(res{2},3)*p]);

        if precondition == 2
            expBglobal = constr_precond_outer( A, X, mu-1, mu );
            res_size = [size(res{1},1), size(res{1},2), size(res{2},2), size(res{2},3), p];
            res_combined = reshape( res_combined, res_size );
            res_combined = apply_global_precond( res_combined, res_size, expBglobal );
        end

        res_combined = reshape( res_combined, [size(res{1},1)*size(res{1},2), size(res{2},2)*size(res{2},3)*p]); 

        [uu,ss,vv] = svd( res_combined, 'econ');
        s = find( diag(ss) > opts.tolOP*norm(diag(ss)), 1, 'last' );
        if opts.maxrankRes ~= 0 
			min( s, opts.maxrankRes );
		end
		res{1} = reshape( uu(:,1:s), [size(res{1},1), size(res{1},2), s]);
        res{2} = reshape( ss(1:s,1:s)*vv(:,1:s)', [s, size(res{2},2), size(res{2},3), p]);

        left = cat(3, X.U{mu-1}, res{1} );
        V = cat(1, X.U{mu}, res{2});

        tmp = permute( V, [1, 4, 2, 3] );
        V = reshape( tmp, [size(tmp,1)*p, size(tmp,3)*size(tmp,4)] );

        [U,S,V] = svd( V, 'econ' );
        s = find( diag(S) > opts.tol*norm(diag(S)), 1, 'last' );
        s = min( s, opts.maxrank );
        V = V(:,1:s)';
        X.U{mu} = reshape( V, [s, sz(2), sz(3)] );

        W = U(:,1:s)*S(1:s,1:s);
        W = reshape( W, [size(tmp,1), p, s]);
        W = permute( W, [1, 3, 2]);
        for k = 1:p
            C{k} = tensorprod( left, W(:,:,k)', 3);
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
                    @(y) apply_local_precond( y, sz, expB), ...
                    max(opts.tolLOBPCG, min( tolLOBPCGmod, sqrt(sum(residuums(:,2*(i-1)+2).^2))/sqrt(p)*tolLOBPCGmod )), ...
                    opts.maxiterLOBPCG, 0);
            else
                [U,L,failureFlag,Lhist ] = lobpcg( rand( prod(sz), p), ...
                    @(y) Afun_block_optim( A, y, sz, left, right, mu), ...
                    opts.tolLOBPCG, opts.maxiterLOBPCG, 0);
            end
            
        catch
            [U,L] = eigs( @(y) Afun_block_optim( A, y, sz, left, right, mu),prod(sz),p,'SM');
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
%         lamX.U{mu} = repmat( reshape(L, [1 1 1 p]), [X.rank(mu), X.size(mu), X.rank(mu+1), 1]).*lamX.U{mu};
%         res = apply(A, X) - lamX;
        
        lamX.U{mu} = bsxfun(@times,lamX.U{mu},reshape(L,[1 1 1 p]));
        Ax.U{mu} = apply(A,X.U{mu},mu);res = Ax - lamX;
        
        
        
        if opts.maxrankRes ~= 0
            res_sz = [size(res.U{mu},1), size(res.U{mu},2), size(res.U{mu},3), size(res.U{mu},4)];
            res.U{mu} = reshape( permute( res.U{mu}, [1 2 4 3]), [ res_sz(1), res_sz(2)*p, res_sz(3)]);
            res = truncate( res, [1 opts.maxrankRes*ones(1,d-1) 1] );
            res.U{mu} = ipermute( reshape( res.U{mu}, [ res.rank(mu), res_sz(2), p, res.rank(mu+1)] ), [1 2 4 3]);
        end
        
        % ugly hack to calc the residual
%         for j = 1:p
%             tmp = res;
%             tmp.U{mu} = tmp.U{mu}(:,:,:,j);
%             resi_norm(j) = norm(tmp);
%             residuums(j,2*(i-1)+2) = resi_norm(j);
%         end
    
        res_o = orthogonalize(res,mu); 
        resi_norm = sqrt(sum(reshape(res_o.U{mu},[],p).^2))';%norm(resi_norm2(:) - resi_norm(:))
        residuums(:,2*(i-1)+2)= resi_norm;
        micro_res = [micro_res, resi_norm];

        if precondition == 1
            res = apply(prec, res);
        end
        res = contract( X, res, [mu, mu+1]);  % res{1}: r{n-1} x In x rn x p
        res_left = permute( res{1}, [1 2 4 3] ); % res{1}: r{n-1} x In x p x rn
        res_left = reshape( res_left, [size(res{1},1)*size(res{1},2)*p, size(res{1},3)]);
        res_combined =  res_left * unfold(res{2},'right');

        if precondition == 2
            expBglobal = constr_precond_outer( A, X, mu, mu+1 );
            res_size = [size(res{1},1), size(res{1},2), p, size(res{2},2), size(res{2},3)];
            res_combined = reshape( res_combined, res_size );
            res_combined = permute( res_combined, [1 2 4 5 3]);  % move p to back
            res_combined = apply_global_precond( res_combined, res_size([1 2 4 5 3]), expBglobal );
            res_combined = ipermute( res_combined, [1 2 4 5 3]);  % move p to back
        end
        
        res_combined = reshape( res_combined, [size(res{1},1)*size(res{1},2)*p, size(res{2},2)*size(res{2},3)]); 

        [uu,ss,vv] = svd( res_combined, 'econ');
        s = find( diag(ss) > opts.tolOP*norm(diag(ss)), 1, 'last' );
		if opts.maxrankRes ~= 0 
			min( s, opts.maxrankRes );
		end
        res{1} = reshape( uu(:,1:s)*ss(1:s,1:s), [size(res{1},1), size(res{1},2), p, s]);
        res{2} = reshape( vv(:,1:s)', [s, size(res{2},2), size(res{2},3)]);

        right = cat(1, X.U{mu+1}, res{2} );
        res{1} = permute( res{1}, [1 2 4 3]);
        U = cat(3, X.U{mu}, res{1});

        tmp = permute( U, [1, 2, 4, 3] );
        U = reshape( tmp, [size(tmp,1)*size(tmp,2), p*size(tmp,4)] );

        [U,S,V] = svd( U, 'econ' );
        s = find( diag(S) > opts.tol*norm(diag(S)), 1, 'last' );
        s = min( s, opts.maxrank );
        U = U(:,1:s);
        X.U{mu} = reshape( U, [sz(1), sz(2), s] );
        W = S(1:s,1:s)*V(:,1:s)';
        W = reshape( W, [s, p, size(tmp,4)]);
        W = permute( W, [1, 3, 2]);
        for k = 1:p
            C{k} = tensorprod( right, W(:,:,k), 1);
        end
        X.U{mu+1} = C{1};

		if opts.verbose
		    disp( ['Augmented system of size (', num2str( [sz(1)*p, sz(2)*sz(3)]), '). Cut-off tol: ', num2str(opts.tol) ])
		    disp( sprintf( 'Number of SVs: %i. Truncated to: %i => %g %%', length(diag(S)), s, s/length(diag(S))*100))
			disp(' ')
		end

    end
    %residuums = [residuums, norm(apply(A,X) - lambda(end)*X)];

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
    evs(i) = innerprod( evec, apply(A, evec));%/ innerprod( evec, evec);
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
    % Tensor contraction between left* (A.U{mu} and U) * right

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
     

function res = apply_local_precond( U, sz, expB)

    p = size(U, 2);

    x = reshape( U, [sz, p] );
    res = zeros( [sz, p] );

    for i = 1:size( expB, 2)
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

function res = apply_global_precond( x, res_sz, expB)

    res = zeros( res_sz );

    for i = 1:size( expB, 2)
        tmp = reshape( x, [res_sz(1), prod(res_sz(2:5))] );
        tmp = reshape( expB{1,i}*tmp, res_sz );

        tmp = reshape( permute( tmp, [2 1 3 4 5] ), [res_sz(2), res_sz(1)*prod(res_sz(3:5))] );
        tmp = ipermute( reshape( expB{2,i}*tmp, [res_sz(2), res_sz(1), res_sz(3:5)] ), [2 1 3 4 5] );

        tmp = reshape( permute( tmp, [3 1 2 4 5] ), [res_sz(3), prod(res_sz([1:2,4:5]))] );
        tmp = ipermute( reshape( expB{3,i}*tmp, [res_sz(3), res_sz([1:2,4:5])] ), [3 1 2 4 5] );

        tmp = reshape( permute( tmp, [4 1 2 3 5] ), [res_sz(4), prod(res_sz([1:3,5]))] );
        tmp = ipermute( reshape( expB{4,i}*tmp, [res_sz(4), res_sz([1:3,5])] ), [4 1 2 3 5] );

        res = res + tmp;
    end
    
end
