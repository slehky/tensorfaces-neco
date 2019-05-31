function res = tensorize( U, mode, d )
    %TENSORIZE Tensorize matrix (inverse matricization).
    %   X = TENSORIZE(U, MODE, D) (re-)tensorizes the matrix U along the 
    %   specified mode MODE into a tensor X of size D(1) x D(2) x D(3). Higher 
    %   dimensions than 3 are not supported. Tensorize is inverse matricization,
    %   that is, X == tensorize( matricize(X, i), i, size(X)) for all modes i.
    %
    %   See also MATRICIZE, TENSORPROD, UNFOLD.
    
    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    % fixed in Sept 2014, Phan Anh Huy
    switch mode
        case 1
            res = reshape( U, d );
        case numel(d)
            res = reshape( transpose(U), d );
        otherwise
            res = reshape(U,d([mode 1:mode-1 mode+1:numel(d)]));
            res = ipermute(res,[mode 1:mode-1 mode+1:numel(d)]);
    end
%     switch mode
%         case 1
%             res = reshape( U, d );
%         case 2 
%             res = ipermute( reshape(U, [d(2), d(1), d(3)]), [2, 1, 3] );
%         case 3 
%             res = reshape( transpose(U), d );
%         otherwise
%             error('Invalid mode input in function matricize')
%     end
end
