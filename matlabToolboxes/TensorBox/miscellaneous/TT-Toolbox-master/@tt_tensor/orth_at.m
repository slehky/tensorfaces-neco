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
    
    %   Adapted from the TTeMPS Toolbox.
        
    Un = x.core(x.ps(pos):x.ps(pos+1)-1);
    if strcmpi(dir, 'left')
        [Q,R] = qr(reshape(Un,[],x.r(pos+1)), 0);
        % Fixed signs of x.U{pos},  if it is orthogonal.
        % This needs for the ASCU algorithm when it updates ranks both
        % sides.
        sR = sign(diag(R));Q = bsxfun(@times,Q,sR');R = bsxfun(@times,R,sR);
        
        % Note that orthogonalization might change the ranks of the core Xn
        % and X{n+1}. For such case, the number of entries is changed
        % accordingly. 
        % Need to change structure of the tt-tensor 
        % pos(n+1)
        %
        % update the core X{n}
        Un = reshape( Q, [x.r(pos), x.n(pos), size(Q,2)] );
        % update the core X{n+1}
        Un2 = R* reshape(x.core(x.ps(pos+1):x.ps(pos+2)-1),x.r(pos+1),[]);
%         x2 = x;
        % Check if rank-n is preserved 
        if size(R,1) == x.r(pos+1)
            x.core(x.ps(pos):x.ps(pos+2)-1) = [Un(:); Un2(:)];
        else 
            ps1 = x.ps(pos)+numel(Un(:));
            ps2 = ps1 + numel(Un2(:));
            x.core(x.ps(pos):ps2-1) = [Un(:); Un2(:)];

            x.core(ps2:x.ps(pos+2)-1) = [];
            
            % update ps 
            if numel(x.ps)>pos+2
                x.ps(pos+3:end) = x.ps(pos+3:end) - (x.ps(pos+2)-ps2);
            end
            x.ps(pos+1) = ps1;
            x.ps(pos+2) = ps2;
            x.r(pos+1) = size(R,1);
             
        end
%         norm(x2-x)
        

    elseif strcmpi(dir, 'right') 
        % mind the transpose as we want to orthonormalize rows
        [Q,R] = qr(reshape(Un,x.r(pos),[])', 0);
        % Fixed signs of x.U{pos},  if it is orthogonal.
        % This needs for the ASCU algorithm when it updates ranks both
        % sides.
        sR = sign(diag(R));Q = bsxfun(@times,Q,sR');R = bsxfun(@times,R,sR);
        
        Un = reshape( Q', [], x.n(pos), x.r(pos+1));
        Un2 = reshape(x.core(x.ps(pos-1):x.ps(pos)-1),[],x.r(pos)) * R';
%         x2 = x;
        if size(R,1) == x.r(pos)
            
            x.core(x.ps(pos):x.ps(pos+1)-1) = Un; % Fix error occuring when U{pos} is a fat matrix. The rank needs to be corrected. Phan Anh Huy, Sept.  2014
            x.core(x.ps(pos-1):x.ps(pos)-1) = Un2;
        else
            
            ps1 = x.ps(pos+1)-numel(Un(:));
            ps2 = ps1 - numel(Un2(:));
            x.core(ps2:x.ps(pos+1)-1) = [Un2(:); Un(:)];

            x.core(x.ps(pos-1):ps2-1) = [];
            
            % update ps 
            x.ps(pos) = x.ps(pos-1)+numel(Un2(:));
            x.ps(pos+1:end) = x.ps(pos+1:end) - (ps2- x.ps(pos-1));
            x.r(pos) = size(R,1);
        end
%         norm(x2-x)
        
    else
        error('Unknown direction specified. Choose either LEFT or RIGHT') 
    end
end
