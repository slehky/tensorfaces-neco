function I = div_op(dx, dy, wx, wy)
%DIV_OP Divergence operator in 2 dimentions
%   Usage:  I = div_op(dx, dy)
%           I = div_op(dx, dy, wx, wy)
%
%   Input parameters:
%         dx    : Gradient along x
%         dy    : Gradient along y
%         wx    : Weights along x
%         wy    : Weights along y
%
%   Output parameters:
%         I     : Output divergence image 
%
%   Compute the 2-dimentional divergence of an image. If a cube is given,
%   it will compute the divergence of all images in the cube.
%
%   Url: http://unlocbox.sourceforge.net/doc//misc/div_op.php

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

if nargin > 2
    dx = dx .* conj(wx);
    dy = dy .* conj(wy);
end

I = [dx(1, :,:) ; dx(2:end-1, :,:)-dx(1:end-2, :,:) ; -dx(end-1, :,:)];
I = I + [dy(:, 1,:) , dy(:, 2:end-1,:)-dy(:, 1:end-2,:) , -dy(:, end-1,:)];

end
