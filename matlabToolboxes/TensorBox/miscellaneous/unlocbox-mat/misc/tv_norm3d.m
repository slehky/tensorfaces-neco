function y = tv_norm3d(u)
%TV_NORM3D 3 Dimentional TV norm
%   Usage:  y = tv_norm(x)
%
%   Input parameters:
%         x     : Input data (3 dimentional matrix)
%   Output parameters:
%         sol   : Norm
%
%   Compute the 3-dimentional TV norm of x
%
%   Url: http://unlocbox.sourceforge.net/doc//misc/tv_norm3d.php

% Copyright (C) 2012-2013 Nathanael Perraudin.
% This file is part of LTFAT version 1.1.97
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
        [dx, dy, dz] = gradient_op3d(u);
    
temp = sqrt(abs(dx).^2 + abs(dy).^2 + abs(dz).^2);
y = sum(temp(:));

end
