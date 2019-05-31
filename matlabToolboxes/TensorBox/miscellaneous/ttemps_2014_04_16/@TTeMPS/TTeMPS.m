classdef TTeMPS
% TTeMPS
%
%   A MATLAB class for representing and manipulating tensors
%   in the TT/MPS format. 
% 

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2014
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

properties( SetAccess = public, GetAccess = public )

    U           % core tensors
    orth = 0;   % Indicates the orthogonalization index (0 if no orth)
                % CURRENTLY NOT USED!

end

% Dependent properties
properties( Dependent = true, SetAccess = private, GetAccess = public )

    rank
    order
    size

end

% Get methods for dependent properties (computed on-the-fly)
methods

    function rank_ = get.rank(x)
        rank_ = cellfun( @(x) size(x,1), x.U(:)');
        if ndims(x.U{end}) > 2
            rank_ = [rank_, size(x.U{end},3)];
        else
            rank_ = [rank_, 1];
        end
    end
   
    function size_ = get.size(x)
        size_ = cellfun( @(y) size(y,2), x.U);
    end

    function order_ = get.order(x)
        order_ = length( x.size );
    end
end


methods( Access = public )

    function x = TTeMPS(varargin)
    %TTEMPS Construct a tensor in TT/MPS format and return TTeMPS object.
    %
    %   X = TTEMPS() creates a zero TT/MPS tensor 
    %
    %   X = TTEMPS(CORES) creates a TT/MPS tensor with core tensors C taken 
    %   from the cell array CORES
    %
    %   X = TTEMPS(CORES, ORTH) creates a TT/MPS tensor with core tensors C 
    %   taken from the cell array CORES. ORTH specifies the position of 
    %   orthogonalization (default = 0, no orthogonalization).
    %

        % Default constructor
        if (nargin == 0)
          
            x = TTeMPS( {0 0 0} );
            return;
          
        elseif (nargin == 1)

            x = TTeMPS( varargin{1}, 0 );
            return;

        elseif (nargin == 2)

            % CAREFUL, add sanity check here
            x.U = varargin{1}(:)';
            x.orth = varargin{2};

        else
            error('Invalid number of arguments.')
        end
    end

    % Other public functions
    y = full( x );
    d = ndims( x );
    x = orth_at( x, pos, dir );
    x = orthogonalize( x, pos );
    x = orthogonalize_upto( x, pos,side );
    z = ttxt(x,Y,mode,side);
    res = innerprod( x, y, dir, upto );
    res = norm( x );
    res = contract( x, y, idx );
    x = truncate( x, r );
    x = uminus( x );
    x = uplus( x );
    z = plus( x, y );
    z = minus( x, y );
    x = mtimes( a, x );
    z = hadamard( x, y, idx );
    z = TTeMPS_to_TT( x );
    disp( x, name );
    display( x );
    
end


methods( Static, Access = private )

    x = subsref_mex( r, n, ind , C);

end
end
