function x = orthogonalize( x, pos )
    %ORTHOGONALIZE Orthogonalize tensor.
    %   X = ORTHOGONALIZE( X, POS ) orthogonalizes all cores of the TTeMPS tensor X
    %   except the core at position POS. Cores 1...POS-1 are left-, cores POS+1...end
    %   are right-orthogonalized. Therefore,
    %
    %   X = ORTHOGONALIZE( X, 1 ) right-orthogonalizes the full tensor,
    %
    %   X = ORTHOGONALIZE( X, X.order ) left-orthogonalizes the full tensor.
    %
    %   See also ORTH_AT.
    
    %   adapted from the TTeMPS Toolbox. 
     

    % left orthogonalization till pos (from left)
    for i = 1:pos-1
        x = orth_at( x, i, 'left' );
    end

    % right orthogonalization till pos (from right)
    for i = x.d:-1:pos+1
        x = orth_at( x, i, 'right' );
    end
end
