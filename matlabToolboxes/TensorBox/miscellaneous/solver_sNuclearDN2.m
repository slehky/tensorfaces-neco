function varargout = solver_sNuclearDN2( A, b, epsilon, mu, x0, z0, opts, varargin )
% Modified of solver_sNuclearBPDN
% Phan Anh Huy, September 26, 2016
% where A is a linear operator of x
% SOLVER_SNUCLEARDN Nuclear norm basis pursuit problem with relaxed constraints. Uses smoothing.
% [ x, out, opts ] = solver_sNuclearBPDN( omega, b, epsilon,mu, X0, Z0, opts )
%    Solves the smoothed nuclear norm basis pursuit problem
%        minimize norm_nuc(X) + 0.5*mu*norm(X-X0,'fro').^2
%        s.t.     ||A * x - b || <= epsilon
%    by constructing and solving the composite dual
%        maximize - g_sm(z)
%    where
%        g_sm(z) = sup_x <z,Ax-b>-norm(x,1)-(1/2)*mu*norm(x-x0)
%    A_omega is the restriction to the set omega, and b must be a vector. The
%    initial point x0 and the options structure opts are optional.
%

% Supply default values
error(nargchk(4,8,nargin));
if nargin < 5, x0 = []; end
if nargin < 6, z0 = []; end
if nargin < 7, opts = []; end
if ~isfield( opts, 'restart' ), 
    opts.restart = 50; 
end


% TODO: see the new linop_subsample.m file
%A = @(varargin)linop_nuclear( n1, n2, nnz, omega_lin, omegaI, omegaJ, varargin{:} );
%Ax = linop_matrix( A, 'R2R');

obj     = prox_nuclear(1);  

prox    = prox_l2(epsilon);

% linear operator
% Aop   = linop_matrix( A, 'R2R' ); % n1 x n2 is size of x
% It  = linop_handles({ [n1,n2],[n1*n2,1] }, vec, mat, 'R2R' );
% Aop   = linop_compose( Aop, It );

[varargout{1:max(nargout,1)}] = tfocs_SCD( obj,{ A, -b}, prox, mu, x0, z0, opts, varargin{:} );
  
% %
% function y = linop_nuclear( n1, n2, nnz, x, mode )
% switch mode,
%     case 0,
%         y = { [n1,n2], [nnz,1] }; % size of 
%     case 1,
%         y =  ;% A * x
%     case 2,
%         y = sparse( omegaI, omegaJ, x, n1, n2 ); % A'*x
% end