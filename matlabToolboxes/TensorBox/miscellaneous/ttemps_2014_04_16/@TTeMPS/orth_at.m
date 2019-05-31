function x = orth_at( x, pos, dir )
    %ORTH_AT Orthogonalize single core.
    %   X = ORTH_AT( X, POS, 'LEFT') left-orthogonalizes the core at position POS 
    %   and multiplies the corresponding R-factor with core POS+1. All other cores
    %   are untouched. The modified tensor is returned.
    %
    %   X = ORTH_AT( X, POS, 'RIGHT') right-orthogonalizes the core at position POS
    %   and multiplies the corresponding R-factor with core POS-1. All other cores
    %   are untouched. The modified tensor is returned.
    %
    %   See also ORTHOGONALIZE.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    if strcmpi(dir, 'left')
        [Q,R] = qr( unfold( x.U{pos}, 'left' ), 0);
        % Fixed signs of x.U{pos},  if it is orthogonal.
        % This needs for the ASCU algorithm when it updates ranks both
        % sides.
        sR = sign(diag(R));Q = bsxfun(@times,Q,sR');R = bsxfun(@times,R,sR);
        x.U{pos} = reshape( Q, [x.rank(pos), x.size(pos), size(Q,2)] );
        x.U{pos+1} = tensorprod( x.U{pos+1}, R, 1); 

    elseif strcmpi(dir, 'right') 
        % mind the transpose as we want to orthonormalize rows
        [Q,R] = qr( unfold( x.U{pos}, 'right' )', 0);
        % Fixed signs of x.U{pos},  if it is orthogonal.
        % This needs for the ASCU algorithm when it updates ranks both
        % sides.
        sR = sign(diag(R));Q = bsxfun(@times,Q,sR');R = bsxfun(@times,R,sR);
        
        x.U{pos} = reshape( Q', [], x.size(pos), x.rank(pos+1)); % Fix error occuring when U{pos} is a fat matrix. The rank needs to be corrected. Phan Anh Huy, Sept.  2014
        x.U{pos-1} = tensorprod( x.U{pos-1}, R, 3); 
        
    else
        error('Unknown direction specified. Choose either LEFT or RIGHT') 
    end
end
