function W = unfold( U, dir )
    %UNFOLD Left/right-unfold a core tensor in MPS or MPO.
    %   W = UNFOLD(U, DIR) unfolds the tensor U in direction DIR, where 
    %   DIR is either 'LEFT' or 'RIGHT' (case insensitive).
    %
    %   See also MATRICIZE, TENSORIZE, TENSORPROD.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    d = size(U);
    % pad with 1 for the last dim (annoying)
    if length(size(U)) == 2
        d = [d, 1];
    end

    if strcmpi(dir, 'left')
        W = reshape( U, [prod(d(1:end-1)), d(end)] );
    elseif strcmpi(dir, 'right')
        W = reshape( U, [d(1), prod(d(2:end))] );
    else
        error('Unknown direction specified. Choose either LEFT or RIGHT') 
    end
end

% function W = unfold( U, dir )
%     %UNFOLD Left/right-unfold a 3D array.
%     %   W = UNFOLD(U, DIR) unfolds the 3-dim. tensor U in direction DIR, where 
%     %   DIR is either 'LEFT' or 'RIGHT' (case insensitive).
%     %
%     %   See also MATRICIZE, TENSORIZE, TENSORPROD.
% 
%     %   TTeMPS Toolbox. 
%     %   Michael Steinlechner, 2013-2014
%     %   Questions and contact: michael.steinlechner@epfl.ch
%     %   BSD 2-clause license, see LICENSE.txt
% 
%     d = size(U);
%     % pad with 1 for the last dim (annoying)
%     if length(size(U)) == 2
%         d = [d, 1];
%     end
% 
%     if strcmpi(dir, 'left')
%         W = reshape( U, [d(1)*d(2), d(3)] );
%     elseif strcmpi(dir, 'right')
%         W = reshape( U, [d(1), d(2)*d(3)] );
%     else
%         error('Unknown direction specified. Choose either LEFT or RIGHT') 
%     end
% end
