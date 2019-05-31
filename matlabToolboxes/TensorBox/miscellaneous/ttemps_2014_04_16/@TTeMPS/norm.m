function res = norm( x )
    %NORM Norm of a TT/MPS tensor.
    %   norm(X) computes the Frobenius norm of the TT/MPS tensor X.
    %
    %   See also INNERPROD
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    x = orthogonalize(x, x.order );
    res = norm( x.U{end}(:) );
    
end
