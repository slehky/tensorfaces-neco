function x = round( x, tol )
    %ROUND Approximate TTeMPS tensor within a prescribed tolerance.
    %   X = ROUND( X, tol ) truncates the given TTeMPS tensor X to a
    %   lower rank such that the error is in order of tol.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt
    
    
    % Left-right procedure
    x = x.orthogonalize( x.order );

    for i = x.order:-1:2
        [U,S,V] = svd( unfold( x.U{i}, 'right'), 0 );
        r = find( diag(S) > tol, 1, 'last' );
        if isempty(r), r = 1;end
        U = U(:,1:r);
        V = V(:,1:r);
        S = S(1:r,1:r);
        x.U{i} = reshape( V', [r, x.size(i), x.rank(i+1)] );
        x.U{i-1} = tensorprod( x.U{i-1}, (U*S)', 3 );
    end

end
