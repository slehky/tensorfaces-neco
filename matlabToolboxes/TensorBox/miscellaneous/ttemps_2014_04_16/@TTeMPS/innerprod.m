function res = innerprod( x, y, dir, upto )
    %INNERPROD Inner product between two TT/MPS tensors.
    %   innerprod(X,Y) computes the inner product between the TT/MPS tensors X and Y.
    %   Assumes that the first rank of both tensors, X.rank(1) and Y.rank(1), is 1. 
    %   The last rank may be different from 1, resulting in a matrix of size 
    %   [X.rank(end), Y.rank(end)].
    %
    %   See also NORM

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.

    % Phan Anh Huy, September 2014.
    % This innerprod is confusing, ans should be named contraction between
    % two two TT/MPS tensors x and x along modes indicated by "upto"
    %
    % When contraction is for all modes, we have innerproduct between x and
    % y
    %
    % This added part is to compute innerproduct between an array and a
    % TT/MPS tensor. This kind of inner product is useful to compute the
    % Frobenius norm between an order-N array and a TT/MPS tensor, without
    % converting the TT-tensor to full form
    %    || X - \tX ||_F^2 = ||X||_F^2 + ||\tX||_F^2 - 2 <X, \tX>
    %
    % norm(X-full(tX))^2 = norm(X)^2 + norm(tX)^2 - 2*innerprod(X,tX)
    % tX: is a TT/MPS tensor.
    
    if isa(x,'TTeMPS') && isa(y,'double')
        res = innerprod(y,x);
    elseif isa(x,'double') && isa(y,'TTeMPS')
        % This routine considers Y.rank(1) = Y.rank(N) = 1
        SzX = size(x);SzY = y.size;
        x = squeeze(y.U{1})'*reshape(x,SzX(1),[]); % r(1)xI(2)x...xI(N)
        x = reshape(x,[],SzY(end))*y.U{end}';    % r(1)xI(2)x...xr(N)
        r = y.rank;
        for k = 2:y.order-2
            x = reshape(y.U{k},[],r(k+1))' * reshape(x,r(k)*SzX(k),[]);
        end
        res = x(:).'*y.U{end-1}(:);
    
    elseif isa(x,'TTeMPS') && isa(y,'TTeMPS')
        if ~exist( 'dir', 'var' )
            dir = 'LR';
        end
        if ~exist( 'upto', 'var' )
            if strcmpi( dir, 'LR')
                upto = x.order;
            else
                upto = 1;
            end
        end
        
        % Left-to-Right procedure
        if strcmpi( dir, 'LR')
            res = unfold( x.U{1}, 'left')' * unfold( y.U{1}, 'left');
            
            for i = 2:upto
                tmp = tensorprod( x.U{i}, res', 1);
                res = unfold( tmp, 'left')' * unfold( y.U{i}, 'left');
            end
            
            % Right-to-Left procedure
        elseif strcmpi( dir, 'RL')
            d = x.order;
            res = conj(unfold( x.U{d}, 'right')) * unfold( y.U{d}, 'right').';
            
            for i = d-1:-1:upto
                tmp = tensorprod( x.U{i}, res', 3);
                res = conj(unfold( tmp, 'right')) * unfold( y.U{i}, 'right').';
            end
            
        else
            error('Unknown direction specified. Choose either LR (default) or RL')
        end
    end
end

