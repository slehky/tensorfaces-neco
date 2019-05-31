function n = norm_nuclear(x)
%NORM_NUCLEAR - Nuclear norm of x
%   Usage: norm_nuclear(x) 
%   
%   return the nuclear norm of x
%
%
%   The input arguments are:
%
%   - x : the matrix which we want the norm
%
%
%   Url: http://unlocbox.sourceforge.net/doc//misc/norm_nuclear.php

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

% Author:  Nathanael Perraudin
% Date: June 2012
%

% [~,Sigma,~] =  svd(x,'econ');
% n = sum(diag(Sigma));
Sigma = svd(x);
n = sum(Sigma);


end
