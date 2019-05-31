function varargout = solver_sNuclearBPDN_2( omega, b, epsilon, mu, x0, z0, opts, varargin )
% SOLVER_SNUCLEARBPDN Nuclear norm basis pursuit problem with relaxed constraints. Uses smoothing.
% [ x, out, opts ] = solver_sNuclearBPDN( omega, b, epsilon,mu, X0, Z0, opts )
%    Solves the smoothed nuclear norm basis pursuit problem
%        minimize norm_nuc(X) + 0.5*mu*norm(X-X0,'fro').^2
%        s.t.     ||A_omega * x - b || <= epsilon
%    by constructing and solving the composite dual
%        maximize - g_sm(z)
%    where
%        g_sm(z) = sup_x <z,Ax-b>-norm(x,1)-(1/2)*mu*norm(x-x0)
%    A_omega is the restriction to the set omega, and b must be a vector. The
%    initial point x0 and the options structure opts are optional.
%
%   The "omega" term may be in one of three forms:
%       (1) OMEGA, a sparse matrix.  Only the nonzero pattern is important.
%       (2) {n1,n2,omega}, a cell, where [n1,n2] = size(X), and omega
%               is the vector of linear indices of the observed set
%       (3) {n1,n2,omegaI,omegaJ}, a cell.  Similar to (2), except the set
%               omega is now specified by subscripts. Specifically,
%               omega = sub2ind( [n1,n2], omegaI, omegaJ) and
%               [omegaI,omegaJ] = ind2sub( [n1,n2], omega )

% Supply default values
error(nargchk(4,8,nargin));
if nargin < 5, x0 = []; end
if nargin < 6, z0 = []; end
if nargin < 7, opts = []; end
if nargin < 8, maxRank = []; else maxRank = varargin{1}; end
if ~isfield( opts, 'restart' ),
    opts.restart = 50;
end

% if isempty(omega)
%     error( 'Sampling operator cannot be empty.' );
% elseif issparse(omega)
%     [omegaI,omegaJ] = find(omega);
%     [n1,n2]         = size(omega);
%     omega_lin       = sub2ind( [n1,n2], omegaI, omegaJ );
% elseif iscell(omega)
%     switch length(omega)
%         case 3,
%             [ n1, n2, omega_lin ] = deal( omega{:} );
%             [omegaI,omegaJ]       = ind2sub( [n1,n2], omega_lin );
%         case 4
%             [ n1, n2, omegaI, omegaJ ] = deal( omega{:} );
%             omega_lin = sub2ind( [n1,n2], omegaI, omegaJ );
%         otherwise
%             error( 'Incorrect format for the sampling operator.' );
%     end
% else
%     error( 'Incorrect format for the sampling operator.' );
% end
% nnz = numel(omega_lin);
[n1,n2] = deal(omega{1:2});
nnz = n1*n2;
if ~isequal( size(b), [ nnz, 1 ] ),
    error( 'Incorrect size for the sampled data.' );
end

% TODO: see the new linop_subsample.m file
A = @(varargin)linop_nuclear( n1, n2, nnz,  varargin{:} );
prox_op = prox_nuclear_2(1,[],maxRank);

% [varargout{1:max(nargout,1)}] = ...
%     tfocs_SCD( prox_op, { A, -b }, prox_l2(epsilon), mu, x0, z0, opts, varargin{:} );
[varargout{1:max(nargout,1)}] = ...
    tfocs_SCD( prox_op, { A, -b }, prox_l2(epsilon), mu, x0, z0, opts);
end

%
% Implements the matrix sampling operator: X -> [X_ij]_{i,j\in\omega}
%
function y = linop_nuclear( n1, n2, nnz, x, mode )
switch mode,
    case 0,
        y = { [n1,n2], [nnz,1] };
    case 1,
        y = x(:);
    case 2,
        y = reshape(x, n1, n2 );
end
end

% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.



function op = prox_nuclear_2( q, LARGESCALE , maxRank)

%PROX_NUCLEAR    Nuclear norm.
%    OP = PROX_NUCLEAR( q ) implements the nonsmooth function
%        OP(X) = q * sum(svd(X)).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    it must be a positive real scalar.
%
%    OP = PROX_NUCLEAR( q, LARGESCALE )
%       uses a Lanczos-based SVD if LARGESCALE == true,
%       otherwise it uses a dense matrix SVD
%
%    CALLS = PROX_NUCLEAR( 'reset' )
%       resets the internal counter and returns the number of function
%       calls
%
% This implementation uses a naive approach that does not exploit any
% a priori knowledge that X and G are low rank or sparse. Future
% implementations of TFOCS will be able to handle low-rank matrices
% more effectively.
% Dual: proj_spectral.m
% See also prox_trace.m  and proj_spectral.m

if nargin == 1 && strcmpi(q,'reset')
    op = prox_nuclear_impl;
    return;
end

if nargin == 0,
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) || numel( q ) ~= 1 || q <= 0,
    error( 'Argument must be positive.' );
end
if nargin < 2, LARGESCALE = []; end

% clear the persistent values:

if ~isempty(maxRank)
    prox_nuclear_impl(maxRank);
else
    prox_nuclear_impl();
end

op = @(varargin)prox_nuclear_impl( q, LARGESCALE, varargin{:} );

end % end of main function

function [ v, X ] = prox_nuclear_impl( q, LARGESCALE, X, t )
persistent oldRank
persistent nCalls
persistent maxRank
if nargin == 0, oldRank = []; v = nCalls; nCalls = []; maxRank = [];return; end
if nargin == 1, maxRank = q; oldRank = []; v = nCalls; nCalls = []; return; end
if strcmp(maxRank,'none') maxRank = [];end
if isempty(nCalls), nCalls = 0; end

ND = (size(X,2) == 1);
% ND = ~ismatrix(X);
if ND, % X is a vector, not a matrix, so reshape it
    sx = size(X);
    X = reshape( X, prod(sx(1:end-1)), sx(end) );
end

if nargin > 3 && t > 0,
    
    if ~isempty(LARGESCALE) % inherited from parent
        largescale = LARGESCALE;
    else
        largescale = ( numel(X) > 100^2 ) && issparse(X);
    end
    tau = q*t;
    nCalls = nCalls + 1;
    
    %     fprintf('ranks: ');
    if ~largescale
        sx = size(X);
        [mxs,modx] = max(sx);
        if ~isempty(maxRank) && maxRank < mxs/2
            [U,S,V] = mysvds(full(X),maxRank);
        else
            if mxs<2000
                
                if modx == 1
                    
                    %%
                    Cx = full(X'*X);mx = max(Cx(:));
                    if any(Cx(:))
                        [V,S] = eig(Cx/mx);
                        S = sqrt(diag(S));
                        U = X*V*diag(1./S);
                        S = diag(S);
                        
                        [Uq,Ur] = qr(U,0);
                        [uu,S,vv] = svd(Ur*S);
                        U = Uq*uu;
                        V = V*vv;
                    else
                        U = zeros(sx);
                        V = zeros(sx(2));
                        S = zeros(sx(2));
                    end
                else
                    
                    %%
                    Cx = full(X*X');mx = max(Cx(:));
                    if any(Cx(:))
                        [U,S] = eig( Cx/mx);
                        S = sqrt(diag(S));
                        V = (U'*X)'*diag(1./S)*1/sqrt(mx);
                        S = diag(S);
                        
                        [Vq,Vr] = qr(V,0);
                        [vv,S,uu] = svd(Vr*S);
                        U = U*uu;
                        V = Vq*vv;
                        
                    else
                        U = zeros(sx(1));
                        V = zeros(sx);
                        S = zeros(sx(1));
                    end
                end
                
            else
                % Guess which singular value will have value near tau:
                [M,N] = size(X);
                if isempty(oldRank), K = 10;
                else, K = oldRank + 2;
                end
                
                ok = false;
                opts = [];
                opts.tol = 1e-10; % the default in svds
                opt  = [];
                opt.eta = eps; % makes compute_int slow
                %         opt.eta = 0;  % makes reorth slow
                opt.delta = 10*opt.eta;
                while ~ok
                    K = min( [K,M,N] );
                    if exist('lansvd','file')
                        [U,S,V] = lansvd(X,K,'L',opt );
                    else
                        [U,S,V] = svds(X,K,'L',opts);
                    end
                    ok = (min(diag(S)) < tau) || ( K == min(M,N) );
                    %             fprintf('%d ',K );
                    if ok, break; end
                    %             K = K + 5;
                    K = 2*K;
                    if K > 10
                        opts.tol = 1e-6;
                    end
                    if K > 40
                        opts.tol = 1e-4;
                    end
                    if K > 100
                        opts.tol = 1e-1;
                    end
                    if K > min(M,N)/2
                        %                 disp('Computing explicit SVD');
                        [U,S,V] = svd( full(X), 'econ' );
                        ok = true;
                    end
                end
                oldRank = length(find(diag(S) > tau));
            end
        end
        s  = diag(S) - tau;
        tt = s > 0;
        s  = s(tt,:);
        
        %     fprintf('\n')';
        %     fprintf('rank is %d\n', length(tt) );
        
        % Check to make sure this doesn't break existing code...
        if isempty(s),
            X(:) = 0;  % this line breaks the packSVD version
            %         X = tfocs_zeros(X);
        else
            X = U(:,tt) * bsxfun( @times, s, V(:,tt)' );
        end
            
    else
        s = svd(full(X)); % could be expensive!
    end
    
    v = q * sum(s);
    if ND,
        X = reshape( X, sx );
    end
    
end
end


% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.