function [U, S, V, flag] = svds(A,varargin)
% SVDS   Find a few singular values and vectors.
%
%  S = SVDS(A) returns the 6 largest singular values of A.
%
%  S = SVDS(A,K) computes the K largest singular values of A.
%
%  S = SVDS(A,K,SIGMA) computes K singular values based on SIGMA:
%
%        'largest' - compute K largest singular values. This is the default.
%       'smallest' - compute K smallest singular values.
%     'smallestnz' - compute K smallest non-zero singular values.
%         numeric  - compute K singular values nearest to SIGMA.
%
%  Note: if more singular values are requested than are available, K is set
%  to the maximum possible number.
%
%  S = SVDS(A,K,SIGMA,OPTIONS) sets some parameters (see EIGS):
%  
%  Field name         Parameter                               Default
%  
%  OPTIONS.tol       Convergence tolerance                      1e-10
%  OPTIONS.maxit     Maximum number of iterations.                100
%  OPTIONS.p         Maximum size of Krylov subspace       max(3*K,15)
%  OPTIONS.u0        Left initial starting vector                  ~
%  OPTIONS.v0        Right initial starting vector                 ~
%  
%  Provide at most one of u0 or v0. If neither is provided, then for an
%  M-by-N matrix A, the default is chosen as follows:
%    M < N    u0 = randn(M,1);
%    M >= N   v0 = randn(N,1);
%  When using numerical SIGMA, OPTIONS.p is ignored.
%
%  [U,S,V] = SVDS(A,...) computes the singular vectors as well.
%  If A is M-by-N and K singular values are computed, then U is M-by-K
%  with orthonormal columns, S is K-by-K diagonal, and V is N-by-K with
%  orthonormal columns.
%  
%  [U,S,V,FLAG] = SVDS(A,...) also returns a convergence flag.
%  If the method has converged, then FLAG is 0. If it did not converge,
%  then FLAG is 1. 
%  
% [...] = SVDS(AFUN, MN, ...) accepts a function handle AFUN instead of 
%  the matrix A. AFUN(X,'notransp') must accept a vector input X and return
%  the matrix-vector product A*X, while AFUN(X,'transp') must return A'*X. 
%  MN is a 1-by-2 vector [m n] where m is the number of rows of A and
%  n is the number of columns of A. Function handles can only be used
%  in the case where SIGMA = 'largest'.
%  
%  Note: SVDS is best suited for finding a few singular values of a large,
%  sparse matrix. To find all the singular values of such a matrix,
%  SVD(FULL(A), 'econ') usually performs better than SVDS(A,MIN(SIZE(A))).
%  
%  Example:
%    C = gallery('neumann',100);
%    sf = svd(full(C))
%    sl = svds(C,10)
%    ss = svds(C,10,'smallest')
%    snz = svds(C,10,'smallestnz')
%    s2 = svds(C,10,2)
%    
%    sl will be a vector of the 10 largest singular values, ss will be a
%    vector of the 10 smallest singular values, snz will be a vector of the
%    10 smallest singular values which are nonzero and s2 will be a
%    vector of the 10 singular values of C which are closest to 2.
%  
%  See also SVD, EIGS.

% This code uses thickly restarted Lanczos bidiagonalization for 
%   sigma = 'largest', 'smallest', or 'smallestnz' and uses EIGS(B,...), 
%   where B = [SPARSE(M,M) A; A' SPARSE(N,N)], when sigma is numeric. 
%
% REFERENCES:
% James Baglama and L. Reichel, Augmented Implicitly Restarted Lanczos 
%    Bidiagonalization Methods, SIAM J. Sci. Comput., 27 (2005), pp. 19-42.
% R. M. Larsen, Lanczos bidiagonalization with partial reorthogonalization,
%    Department of Computer Science, Aarhus University, Technical report, 
%    DAIMI PB-357, September 1998.

% Initialize a dedicated randstream, to make output reproducible.
randStr = RandStream('mt19937ar','Seed',0);

VarIn = varargin;
%%% Get Inputs and do Error Checking %%%
[A,m,n,k,sigma,u0,v0,InnerOpts,Options] = CheckInputs(A,VarIn,randStr);

% Reset the stream
reset(randStr, 1);

%%% Quick return for empty matrix or k = 0 %%%
if k == 0    
    if nargout <= 1
        U = zeros(0,1);
    else
        U = eye(m,0);
        S = zeros(0,0);
        V = eye(n,0);
        flag = 0;
    end

%%% Case Sigma is numeric --- Call old algorithm %%%

elseif isnumeric(sigma) 
    
    oldRandStr = RandStream.setGlobalStream(randStr);
    
    cleanupObj = onCleanup(@() RandStream.setGlobalStream(oldRandStr));
    
    % Note that we pass in 'Options' exactly as given, or as an empty struct
    if nargout <= 1
        U = matlab.internal.math.svdsUsingEigs(A,k,sigma,Options);
    else
        [U,S,V,flag] = matlab.internal.math.svdsUsingEigs(A,k,sigma,Options);
    end

%%% Case Sigma is 'largest' %%%
elseif isequal(sigma,'largest')
    
    if InnerOpts.p >= min(m, n)
        % Since we are going to build the whole subspace anyway, just do a
        % full SVD now. Ignore starting vector and InnerOpts.
        if isa(A, 'function_handle')
            Amat = getMatrix(A, m, n);
        else
            Amat = full(A);
        end
        [U, S, V, flag] = fullSVD(Amat, k, m, n, sigma);
        
        if nargout <= 1
           U = diag(S); 
        end
        return;
    end
    
    % Use the function handle if given, otherwise build it
    if isa(A, 'function_handle')
        Afun = A;
    else
        Afun = @(x,transflag)AfunL(A,x,transflag);
    end
    
    % Exactly one of u0 and v0 is empty, determine which it is
    if ~isempty(v0)
        v = v0;
        f1 = 'notransp';
        f2 = 'transp';
    else
        v = u0;
        % If we start with u0, we essentially pass A' into Lanczos
        f1 = 'transp';
        f2 = 'notransp';
        mtemp = m;
        m = n;
        n = mtemp;
    end
    
    % Normalize starting vector
    v = v/norm(v);
    
    % Check that normalized vector has norm 1
    if ~(abs(norm(v) - 1) < 1e-14)
        error(message('MATLAB:svds:InvalidStartingVector'))
    end
     
    % Call Lanczos bidiagonalization
    [U,S,V,flag] = LanczosBD(Afun,m,n,f1,f2,k,v,InnerOpts,randStr);
    
    if nargout < 2
        % Just return a vector of singular values
        U = diag(S);
    elseif ~isempty(u0)
        % If we used u0, U and V are sing. vectors of A', so we swap them
        Utemp = U;
        U = V;
        V = Utemp;
    end
    
    % If flag is not returned, give a warning about convergence failure
    if nargout < 4 && flag ~= 0
        warning(message('MATLAB:svds:PartialConvergence'));
    end

%%% Case Sigma is 'smallest' or 'smallestnz' %%%
elseif isequal(sigma, 'smallest') || isequal(sigma, 'smallestnz')
    %%% First do a QR factorization of A, and apply solver to inverse of R %%%
    
    % Need to compute non-zero singular values and vectors
    ComputeNonZeroSV = true;
    % Work on the transposed the problem
    transp = n > m;
    % Keep original size of A, use these variables internally
    mIn = m;
    nIn = n;
    
    if transp
        % Transpose before QR so R is square
        A = A';
        % Switch starting vectors u0 and v0
        tmp = u0;
        u0 = v0;
        v0 = tmp;
        % Save new size of A in the internal m and n
        nIn = m;
        mIn = n;
    end
    
    % Do the QR factorization.
    % Q1func: function handle that applies Q1*x or Q1'*x
    % sizeR: number of columns of R that contain nonzeros
    % Q1end: columns of Q1 lying in the null-space of A (only for dense A)
    [Q1func, R1, perm1, sizeR, Q1end] = implicitQR(A);
        
    % Number of Singular values that are exactly zero
    zeroSing = nIn - sizeR;
    
    % Do we need to do a second QR factorization
    DoaSecondQR = zeroSing ~= 0;
    
    % Remove all zero rows (also ensures R1 is 'economy' sized when sparse)
    R1 = R1(1:sizeR,:);
    
    if DoaSecondQR
        % R1 is now broad instead of square, so we do a second QR
        % decomposition to make sure that we have a square, nonsingular
        % matrix to pass into the Lanczos process
        if  issparse(A) 
            % Here we reset the internal tolerance for sparse qr so that it
            % does not set small values to zero. We do not want to send an
            % explicitly singular matrix into the Lanczos process.
            currentQRtol = spparms('spqrtol');
            cleanup1 = onCleanup(@()spparms('spqrtol',currentQRtol));
            spparms('spqrtol',0);            
        end
        
        % Do the second qr
        [Q2func, R2, perm2, ~, Q2end] = implicitQR(R1', sizeR);
        
        if isequal(sigma, 'smallest')
            
            if zeroSing >= k
                % If all the requested singular values are zero, we skip
                % the Lanczos process
                ComputeNonZeroSV = false;
                zeroSing = k;
                
            else
                % Reduce k since we do not need to find all the singular
                % values with the Lanczos process, just the nonzeros
                k = k - zeroSing;
            end
            
        else % Case smallestnz
            if k > sizeR
                % We can only find at most sizeR singular values through
                % the Lanczos process, so we must cap k there
                k = sizeR;
                if k == 0
                    % Occurs only when sizeR is zero
                    ComputeNonZeroSV = false;
                end
            end
            % No singular vectors for singular value 0 need to be computed
            zeroSing = 0;
        end
    end
    
    % Compute non-zero singular values and vectors
    if ComputeNonZeroSV
        
        if InnerOpts.p < sizeR
            % Use the Lanczos process
            
            % Create variable v for starting vector
            if ~isempty(u0)
                v = u0;
            else
                v = v0;
            end
            
            % Normalize starting vector.
            v = v/norm(v);
            
            % Check that the normalized vector indeed has norm 1
            if ~(abs(norm(v) - 1) < 1e-14)
                error(message('MATLAB:svds:InvalidStartingVector'))
            end
            
            % Preprocess the vector v to get it into the correct space
            % (e.g. span of R2 instead of span of A)
            if ~isempty(v0)
                v = v(perm1,:);
                if DoaSecondQR
                    v = Q2func(v, 'transp');
                end
            else
                v = Q1func(v, 'transp');
                if DoaSecondQR
                    v = v(perm2,:);
                end
            end
            
            % Renormalize (in case of numerical error in Q1func and Q2func)
            v = v / norm(v);
            
            % Build Function Handle to pass into Lanczos.
            % Problem flips if we do a second QR (since R1' = Q2*R2 )
            if DoaSecondQR
                Afun = @(X,transflag)AfunS(R2,X,transflag);
                f1 = 'notransp';
                f2 = 'transp';
            else
                Afun = @(X,transflag)AfunS(R1,X,transflag);
                f1 = 'transp';
                f2 = 'notransp';
            end
            
            % Do Lanczos Bidiagonalization Process
            [U,S,V,flag] = LanczosBD(Afun,sizeR,sizeR,f1,f2,k,v,InnerOpts,randStr);
            
            % Invert and flip Ritz values
            S = diag(flip(1./diag(S)));
            U = flip(U,2);
            V = flip(V,2);
            
            % If ratio between largest and smallest singular value is
            % large, check that the residuals are acceptable
            checkResiduals = S(1, 1) / S(end, end) > 1e8;
            
        else % InnerOpts.p >= min(m, n)
            
            % Compute a full SVD, since we are going to build the whole
            % subspace anyway. Ignore starting vector v in this case.
            if DoaSecondQR
                [V, S, U, flag] = fullSVD(R2, k, sizeR, sizeR, 'smallest');
            else
                [U, S, V, flag] = fullSVD(R1, k, sizeR, sizeR, 'smallest');
            end
            checkResiduals = false;
            
        end
        
        % Build Ritz vectors for A from the Ritz vectors for R
        if nargout > 1 || checkResiduals
            if DoaSecondQR
                U(perm2,:) = U;
                V = Q2func(V, 'notransp');
            end
            U = full(Q1func(U, 'notransp'));
            V(perm1,:) = V;
        end
        
        if checkResiduals
            % If matrix is badly conditioned, the numerical error in the
            % QR decomposition and inversion may mean that the residuals
            % with A are much larger than the residuals with R^(-1)
            res1 = A*V - U*S;
            res2 = A'*U - V*S;
            maxres = max(sum(conj(res1).*res1, 1), sum(conj(res2).*res2, 1))';
            
            % Only warn in cases where the residual is much worse than tol
            if ~all(maxres < 1e3*InnerOpts.tol*diag(S))
                estCondition = S(1, 1) / S(end, end);
                warning(message('MATLAB:svds:BadResidual', num2str(estCondition, '%g')));
            end
        end
        
    else % No non-zero singular values to be computed
        % Make correctly sized empty U,S,V
        U = zeros(mIn,0);
        V = zeros(nIn,0);
        S = [];
        flag = 0;
    end
    
    if nargout <= 1
        % Just return a vector of singular values; append zeros to account
        % for exactly zero singular values
        U = [diag(S); zeros(zeroSing,1)];
        
    else  % Finish preparing vectors
        
        if zeroSing > 0
            % Append zero singular values to S (which may be empty)
            S = diag([diag(S); zeros(zeroSing,1)]);
            
            if issparse(A)
                % Find vectors in the nullspace of A. These are left singular
                % vectors corresponding to the zero singular values
                Uzero = full(spdiags(ones(zeroSing, 1), -sizeR, mIn, zeroSing));
                Uzero = Q1func(Uzero, 'notransp');
                
                % Find vectors in the nullspace of A'. These are right singular
                % vectors corresponding to the zero singular values
                Vzero = full(spdiags(ones(zeroSing, 1), -sizeR, nIn, zeroSing));
                Vzero = Q2func(Vzero, 'notransp');
                Vzero(perm1, :) = Vzero;
                
            else
                Uzero = Q1end(:,end-zeroSing+1:end);
                Vzero = Q2end(:,end-zeroSing+1:end);
                Vzero(perm1, :) = Vzero;
            end
            
            % Append these to known left singular vectors
            U = [U, Uzero];
            
            % Append these to known right singular values
            V = [V, Vzero];
        end
        
        % If we did the problem on A', switch U and V back
        if transp
            Utemp = U;
            U = V;
            V = Utemp;
        end
        
    end
    
    % If flag is not returned, give a warning about convergence failure
    if nargout < 4 && flag~=0
        warning(message('MATLAB:svds:PartialConvergence'));
    end
    
end

end

%%% Get and error check inputs %%%

function [A,m,n,k,sigma,u0,v0,InnerOpts,Options] = CheckInputs(A,VarIn,randStr)

% Get A and MN
if isa(A, 'function_handle')
    
    if numel(VarIn) < 1
        error(message('MATLAB:svds:InvalidMN'))
    end
    % Variable MN is equal to size(A)
    MN = VarIn{1};
    
    % Error Check m and n
    if ~isPosInt(MN) || ~isrow(MN) || length(MN) ~= 2 || ~all(isfinite(MN))
        error(message('MATLAB:svds:InvalidMN'));
    else
        m = MN(1);
        n = MN(2);
    end
    
    % Remove MN from VarIn. The remaining entries are k, sigma, and Options
    % which matches VarIn when A is not given as a function handle
    VarIn(1) = [];
    
elseif ismatrix(A) && isa(A, 'double')
    
    % Save size of A in m and n
    [m, n] = size(A);
    
else
    error(message('MATLAB:svds:InvalidA'))
end

% VarIn should now be (at most) {k, sigma, Options}
if length(VarIn) > 3  
    error(message('MATLAB:maxrhs'));
end

% Get k
if length(VarIn) < 1
    k = 6;
else
    k = VarIn{1};
    if ~isPosInt(k) || ~isscalar(k)
        error(message('MATLAB:svds:InvalidK'))
    end
end

% Duplicating old behavior, allow k larger than size of matrix
k = min([m,n,k]);

% Get sigma
if length(VarIn) < 2
    sigma = 'largest';
else
    sigma = VarIn{2};
    
    % Error Check sigma
    
    if ischar(sigma)
        ValidSigmas = {'largest','smallest','smallestnz'};
        % j is the index for the correct valid sigma
        j = find(strncmpi(sigma,ValidSigmas, max(length(sigma),1)),1);
        if isempty(j)
            error(message('MATLAB:svds:InvalidSigma'))
        else
            % Reset sigma to the correct valid sigma for cheaper checking
            sigma = ValidSigmas{j};
        end
    elseif isa(sigma, 'double')
        % We pass numeric sigma into old code
        if ~isreal(sigma) || ~isscalar(sigma) || ~isfinite(sigma)
            error(message('MATLAB:svds:InvalidSigma'))
        end
    else
        error(message('MATLAB:svds:InvalidSigma'))
    end
    
    % Function handle is not implemented when sigma is  'smallest', 
    % 'smallestnz', or numeric 
    if isa(A, 'function_handle') && ~isequal(sigma,'largest')
        error(message('MATLAB:svds:FhandleNoLargest'))
    end
    
end

% Defaults for options

InnerOpts = struct;
% Tolerance is used as stopping criteria in Lanczos Process
InnerOpts.tol = 1e-10;
% Maxit is used as stopping criteria in Lanczos Process
InnerOpts.maxit = 100;
% p is the size of the Krylov Subspace
InnerOpts.p = max(3*k,15);

% Left and right starting vector, algorithm can use at most one
u0 = [];
v0 = [];

% Initialize Options as empty struct
Options = struct;

% Get Options, if provided 
if length(VarIn) >= 3
    Options = VarIn{3};
    if isstruct(Options)
        if isfield(Options,'tol')
            InnerOpts.tol = Options.tol;
            if ~isnumeric(InnerOpts.tol) || ~isscalar(InnerOpts.tol) ...
                    || ~isreal(InnerOpts.tol) || ~(InnerOpts.tol >= 0) 
                error(message('MATLAB:svds:InvalidTol'))
            end
        end
        if isfield(Options,'maxit')
            InnerOpts.maxit = Options.maxit;
            if ~isPosInt(InnerOpts.maxit) || ~isscalar(InnerOpts.maxit)...
                    || InnerOpts.maxit == 0
                error(message('MATLAB:svds:InvalidMaxit'))
            end
        end
        if isfield(Options, 'p')
            InnerOpts.p = Options.p;
            if ~isPosInt(InnerOpts.p) || ~isscalar(InnerOpts.p) 
                error(message('MATLAB:svds:InvalidP'))
            end
            if InnerOpts.p < k+2
                error(message('MATLAB:svds:PlessK'))
            end
            
        end
        if isfield(Options, 'v0')
            v0 = Options.v0;
            if ~iscolumn(v0) || length(v0) ~= n || ...
                    ~isa(v0, 'double')
                error(message('MATLAB:svds:InvalidV0'))
            end
        end
        if isfield(Options, 'u0')
            u0 = Options.u0;
            if ~isempty(v0)
                error(message('MATLAB:svds:BothU0andV0'))
            elseif ~iscolumn(u0) || length(u0) ~= m || ...
                    ~isa(u0, 'double')
                error(message('MATLAB:svds:InvalidU0'))
            end
        end
    else
        error(message('MATLAB:svds:Arg4NotOptionsStruct'))
    end
end

if isnumeric(sigma) && sigma == 0
    warning(message('MATLAB:svds:SigmaZero'))
end

% If the user does not provide a starting vector, we start with a random
% starting vector on the smaller side.
if isempty(u0) && isempty(v0)
    if n > m
        u0 = randn(randStr,m,1);
    else
        v0 = randn(randStr,n,1);
    end
end

end

function  [tf] = isPosInt(X)
% Check if X is a non-negative integer
tf = isnumeric(X) && isreal(X) && all(X(:) >= 0) && all(fix(X(:)) == X(:));
end
%%% Functions for function handles %%%

function b = AfunL(A,X,transflag)
% Afun for mode 'largest'

if strcmpi(transflag,'notransp')
    b = A*X;
elseif strcmpi(transflag,'transp')
    b = A'*X;
end
b = full(b); % otherwise, if A is a sparse vector, b is also sparse
end

function b = AfunS(R,X,transflag)
% Afun for mode 'smallest' or 'smallestnz'
 
if strcmpi(transflag,'notransp')
    b = R\X;
elseif strcmpi(transflag,'transp')
    b = R'\X;
end
end

function [U,S,V,flag] = LanczosBD(Afun,m,n,f1,f2,k,v,InnerOpts,randStr)
% Computes the Lanczos Bidiagonalization using Afun

%%% Initialize Lanczos Process %%%

% Initial step to build u, the current left vector
u = Afun(v,f1);

% Error if user-provided function handle does not return an m-by-1 column vector
if ~iscolumn(u) || ~isa(u, 'double') || length(u) ~= m 
    error(message('MATLAB:svds:InvalidFhandleOutput', f1, m));
end

% After the first application of Afun, turn off possible additional 
% warnings for near singularity
W = warning;
cleanup2 = onCleanup(@()warning(W));
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:illConditionedMatrix');
warning('off', 'MATLAB:singularMatrix');

% Normalize and store u
normu = norm(u);
if normu == 0
    % This only happens if v is in the nullspace of A, in which case we
    % randomly restart the algorithm.
    u = randn(randStr,m,1);
    u = u/norm(u);
else
    u = u/normu;
end

% Count how many iterations we have done (that is, how many times we have
% built U and V up to k columns). nrIter is increased on every call to
% LanczosBDInner.
nrIter = 0;

% Keep track of singular values computed in the last run.
svOld = [];

% Build initial matrices U, V and B
U = u;
V = v;
B = normu;

% Call the bidiagonal Lanczos with these starting values
[U,S,V,nconv,nrIter] = LanczosBDInner(Afun, U, B, V, k, ...
    m, n, f1, f2, InnerOpts, nrIter, randStr);

% Check if all k residuals have converged
if nconv < k
    flag = 1;
    return;
end

% Restart the bidiagonal Lanczos method for k+1 singular values, until the
% first k singular values have also converged. This is particularly useful
% for cases of multiple singular values, where LanczosBDInner may skip a
% multiple of a larger singular value in favor of a smaller singular value.
for nrRestartAfterConverge=1:k+2
    % Note on stopping conditione: have not observed needing more than
    % k+1, set to k+2 for safety.
    
    svNew = diag(S);
    
    % Approximation of the  2-norm of A
    Anorm = svNew(1);
    
    % Break loop if singular values have converged
    if ~isempty(svOld)
        % Vector of difference between iterations in each singular value
        changeSV = max(abs(svOld(1:k) -svNew(1:k)));

        if changeSV <= InnerOpts.tol*Anorm
            % We are satisfied and break the loop
            flag = 0;
            return;
        end
    end
    svOld = svNew;
    
    % Break loop if maximum number of iterations is reached
    if nrIter >= InnerOpts.maxit
        flag = 1;
        return;
    end
    
    %%% Initialize a random restart for next Lanczos iteration %%%
    
    % New vector v, orthogonal to converged singular vectors in V
    v = reorth(V, randn(randStr, size(V, 1), 1), m, n, Anorm, randStr);
    
    % Find next u
    Av = Afun(v,f1);
    u = Av - U*(U'*(Av));
    
    % Reorthogonalize (random restart may occur here if A has numeric rank k)
    [u,normu] = reorth(U,u,m,n,Anorm, randStr);
    
    % Initialize B,U, and V for Lanczos restart
    B = blkdiag(S, normu);
    U = [U, u]; %#ok<AGROW>
    V = [V, v]; %#ok<AGROW>

    % Start Lanczos process to find largest k+1 singular values, building
    % Krylov subspaces orthogonal to the columns of U, V respectively.
    [U,S,V,nconv,nrIter] = LanczosBDInner(Afun, U, B, V, k+1, ...
        m, n, f1, f2, InnerOpts, nrIter, randStr);

    U = U(:, 1:k);
    V = V(:, 1:k);
    S = S(1:k, 1:k);

    % Check if all k+1 residuals have converged
    if nconv < k+1
        flag = 1;
        return;
    end
        
end

% Maximum number of restarts was reached. Not observed in practice yet,
% this is just for safety
flag = 1; % unsuccessful return

end

function [U,S,V,nconv,nrIter] = LanczosBDInner(Afun, U, B, V, k, ...
                                    m, n, f1, f2, InnerOpts, nrIter, randStr)
% Computes the Lanczos Bidiagonalization with thick restart of the matrix
% represented by Afun

% If k is less than 5, retain kplus = 5 largest Ritz values and vectors on
% the thick restart, instead of just k.
kplus = min(max(k, 5), InnerOpts.p-1);

% If this is not the first call, retrieve largest singular value from
% previous run from B.
if ~isscalar(B)
   Bnorm = B(1, 1); 
end

% Initialize u, v and normu:
u = U(:, end);
v = V(:, end);
normu = B(end, end);

%%% Here we start the iterations %%%
% Note in initialization we have done the first half-step of the first
% iteration
while nrIter < InnerOpts.maxit
    
    % Increase counter for the next iteration
    nrIter = nrIter + 1;
    
    % Lanczos Steps 
    for j = size(B, 1):InnerOpts.p
        
        % We need an initial approximation of the norm of the matrix Afun
        % represents. This is approximated by the norm of B. We will
        % calculate norm(B) here only in the first iteration, and only in
        % the first few (6) Lanczos steps.
        if nrIter == 1 && j <= 6
            Bnorm = norm(B);
        end
        
        % Find next v
        if nrIter == 1 && j == 1
            % The first time we use Afun with f2, we want to check that the
            % output is the correct size.
            vtemp = Afun(u,f2);
            if ~iscolumn(vtemp) || ~isa(vtemp, 'double') || length(vtemp) ~= n 
                error(message('MATLAB:svds:InvalidFhandleOutput', f2, n));
            end
            
            v = vtemp - v*normu;
        else
            % Every other time, we simply do this calculation from the
            % basic Lanczos bidiagonalization algorithm
            v = Afun(u,f2) - v*normu;
        end
        
        % Reorthogonalize (Note there may be a random restart here)
        [v,normv] = reorth(V,v,m,n,Bnorm,randStr);
        
        if j < InnerOpts.p
            % Stop half-way through the last Lanczos step since we did the 
            % first half step of the iteration before this for-loop
            
            % Store v in V
            V = [V,v]; %#ok<AGROW>
            
            % Find next u 
            u = Afun(v,f1) - u*normv;
            
            % Reorthogonalize (Note there may be a random restart here)
            [u,normu] = reorth(U,u,m,n,Bnorm,randStr);
            
            % Store u
            U = [U,u]; %#ok<AGROW>
            
            % Store the norms in B
            B(end,end + 1) = normv; %#ok<AGROW>
            B(end + 1, end) = normu; %#ok<AGROW>
            
        end
    end
    
    % Check B, U, and V for NaNs and Infs. Note this rarely occurs because
    % of the random restarts
    if ~all(isfinite(B(:))) || ~all(isfinite(U(:))) || ~all(isfinite(V(:)))
        error(message('MATLAB:svds:BadCondition'))
    end
    
    % Find svd of B 
    [Uin,Sin,Vin] = svd(B,0);
    
    % Calculate k Ritz values/vectors
    U = U*Uin(:,1:kplus);
    V = V*Vin(:,1:kplus);
    S = Sin(1:kplus,1:kplus);
    
    % Save approximate 2-norm of A
    Bnorm = S(1,1);
    
    % Find convergence bounds (and augmentation for restart)
    % Note that 'bounds' approximates U'*A - S*V'
    % For an explanation, see the Baglama paper from the References section
    bounds = (normv*Uin(end,1:kplus))';
    
    % Find the number of converged singular values   
    isConverged = abs(bounds(1:k)) <= InnerOpts.tol*diag(S(1:k, 1:k));
    firstNonConv = find(~isConverged, 1, 'first');
    if isempty(firstNonConv)
        nconv = k;
    else
        nconv = firstNonConv - 1;
    end
    
    if nconv >= k
        % Break loop if singular values have converged
        break;
    elseif nrIter == InnerOpts.maxit
        % Break loop if maximum number of iterations is reached
        break
    end
    
    %%% Initialize Lanczos for restart %%%
    % This is the first half-step for the next iteration
    
    % Find next u
    Av = Afun(v,f1);
    u = Av - U*(U'*(Av));
    
    % Reorthogonalize (random restart may occur here)
    [u,normu] = reorth(U,u,m,n,Bnorm,randStr);
    
    % Initialize B,U, and V for Lanczos restart
    B = [S, bounds; zeros(1,kplus), normu];
    U = [U, u]; %#ok<AGROW>
    V = [V, v]; %#ok<AGROW>
    
end

% Restrict U, V and S to largest k Ritz values
if kplus > k
    U = U(:,1:k);
    V = V(:,1:k);
    S = S(1:k,1:k);
end

end


function [y,normy] = reorth(X,y,m,n,Bnorm,randStr)
% Orthogonalize y against X and do a random restart if new y is small

% Reorthogonalize
y = y - X*(X'*y);

% Find norm(y)
normy = norm(y);

if normy <= max(m,n)*Bnorm*eps
    % Attempt to find another Krylov subspace
    normy = 0;
    y = randn(randStr,size(y,1),1);
    y = y - X*(X'*y);
    y = y/norm(y);
else
    y = y/normy;
end
end


function [Qfunc, R, perm, sizeR, Qend] = implicitQR(A, sizeR)
% Compute the QR decomposition of A, and return a function handle that
% applies Q (which is much cheaper than explicitly storing Q for large and
% sparse A). Also computes sizeR, the number of non-zero columns in R.

if issparse(A)
    % Compute QR decomposition, and return Householder vectors and
    % coefficients to represent Q. 
    [H, tau, pinv, R, perm] = matlab.internal.math.implicitSparseQR(A);
else
    if nargin == 1
        [Q,R,perm] = qr(A,0);
    else
        % If sizeR is already defined, cut R back to first sizeR columns
        [Q,R,perm] = qr(A, 'vector');
        R = R(1:sizeR, :);
    end
end

% If sizeR was not defined yet, detect it from R
if nargin == 1
    % Explicitly check for singularity by finding the last nonzero on
    % the diagonal of R
    sizeR = find(diag(R),1,'last');
    if isempty(sizeR)
        % Occurs only when R is all zeros
        sizeR = 0;
    end
end

% Initialize function handle applying Q and Q' as needed.
if issparse(A)
    Qfunc = @(x, transflag) sparseQHandle(sizeR, ...
        H, tau, pinv, x, isequal(transflag, 'transp'));
    Qend = []; % not used in sparse case
else
    Qend = Q(:,sizeR+1:end);
    Q = Q(:, 1:sizeR);
    Qfunc = @(x, transflag) AfunL(Q, x, transflag);
end

end


% Applies Q(:, 1:sizeR)*x or Q(:, 1:sizeR)'*x, using Householder vectors
% representing Q
function y = sparseQHandle(sizeR, H, tau, pinv, x, transp)
% The built-in applies full mxm matrix Q. We need to pad with zeros or
% truncate to get the required dimensions.

if ~transp
    x(end+1:size(H, 1), :) = 0;
end

if isreal(H) && isreal(tau) && ~isreal(x)
    % Real Q applied to complex x not supported in built-in
    y = matlab.internal.math.applyHouseholder(H, tau, pinv, real(x), transp) + ...
        1i*matlab.internal.math.applyHouseholder(H, tau, pinv, imag(x), transp);
else
    y = matlab.internal.math.applyHouseholder(H, tau, pinv, x, transp);
end

if transp
    y = y(1:sizeR, :);
end

end


% Extract the underlying matrix of the function handle, for cases where we
% call svd directly.
function Amat = getMatrix(Afun, m, n)

    % Extract Atranspose if this needs fewer applications of Afun
    if m >= n
        mIn = m;
        nIn = n;
        transflag = 'notransp';
    else
        mIn = n;
        nIn = m;
        transflag = 'transp';
    end
    
    Amat = zeros(mIn, nIn);
    I = eye(nIn);
    
    % Check dimensions for first column of Amat
    vec = Afun(I(:, 1), transflag);
    if ~iscolumn(vec) || ~isa(vec, 'double') || length(vec) ~= mIn
        % Will only error if the user provides a function handle that does not
        % output a vector of the expected size
        error(message('MATLAB:svds:InvalidFhandleOutput', transflag, mIn));
    end
    
    % Fill in Amat with function handle output
    Amat(:, 1) = vec;
    for ii=2:nIn
        Amat(:, ii) = Afun(I(:, ii), transflag);
    end
    
    % Transpose Amat if necessary
    if m < n
        Amat = Amat';
    end

end

% Compute svds(Amat, ...) using svd(full(Amat))
function [U, S, V, flag] = fullSVD(Amat, k, m, n, sigma)
    
    % Check all values are finite
    if ~all(isfinite(Amat(:)))
        error(message('MATLAB:svds:BadCondition'))
    end
    
    % Call QR, then svd for inner matrix R
    if ~issparse(Amat) || m == n
        % Direct call to svd
        [U, S, V] = svd(full(Amat), 'econ');
    else
        % Make Amat tall
        if m < n
            Amat = Amat';
        end
        
        % Apply SVD only to R matrix of the QR decomposition of A
        [H, tau, pinv, R, perm] = matlab.internal.math.implicitSparseQR(Amat);
        [U, S, V] = svd(full(R));
        U(end+1:size(A, 1), :) = 0;
        U = matlab.internal.math.applyHouseholder(H, tau, pinv, U, false);
        V(perm, :) = V;
        
        % Revert transposition if needed
        if m < n
            tmp = U;
            U = V;
            V = tmp;
        end
        
    end
    
    % Extract largest or smallest k singular values
    nrSV = min(m, n);
    if isequal(sigma, 'largest')
        ind = 1:k;
    else % case sigma = 'smallest'
        ind = nrSV-k+1:nrSV;
    end
    
    V = V(:, ind);
    U = U(:, ind);
    S = S(ind, ind);
    flag = 0;
    
end
