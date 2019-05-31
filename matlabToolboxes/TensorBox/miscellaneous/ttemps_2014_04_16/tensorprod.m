function res = tensorprod( U, A, mode )
    %TENSORPROD Tensor-times-Matrix product. 
    %   A = TENSORPROD(U, A, MODE) performs the mode-MODE product between the
    %   tensor U and matrix A. Higher dimensions than 3 are not supported.
    %
    %   See also MATRICIZE, TENSORIZE, UNFOLD.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    d = size(U);
    % pad with 1 for the last dim (annoying)
    if length(d) == 2
        d = [d, 1];
    end
    d(mode) = size(A,1);
    
    res = A * matricize( U, mode );
    res = tensorize( res, mode, d );


end
