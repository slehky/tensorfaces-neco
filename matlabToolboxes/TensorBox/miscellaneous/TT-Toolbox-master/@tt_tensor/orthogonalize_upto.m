function x = orthogonalize_upto( x, pos,side )
% Orthogonalize a TT-tensor x upto mode "pos" from left-to-right or
% right-to-left specified by side
%
%   See also ORTH_AT.
%

switch side
    case 'left'
        % left orthogonalization upto pos (from left)
        for i = 1:pos-1
            x = orth_at( x, i, 'left' );
        end
        
    case 'right'
        
        % right orthogonalization upto pos (from right)
        for i = x.d:-1:pos+1
            x = orth_at( x, i, 'right' );
        end
end

end



