function [] = imagescgray( im,nfig,tit,subplot )
%IMAGESCGRAY Display an image in gray
%   Usage: imagescgray(im);
%          imagescgray(im,nfig);
%          imagescgray(im,nfig,title);
%          imagescgray(im,nfig,title,subplot);
%
%   Input parameters:
%         Im     : Image in matrix form
%         nfig   : Number of the figure 
%         tit  : Title of the image (string)
%         subplot: Number of the subplot
%   Output parameters:
%
%   This function display an image in gray on this form:
%   
%        figure(nfig);
%        subplot(subplot)
%        imagesc(im);
%        colormap gray;
%        hold on;
%        title(tit);
%        axis off;          
%        axis image;
%        drawnow;
%
%   If nfig=0, then function will automatically create a new figure
%   using:
%        
%        figure();
%
%
%   Url: http://unlocbox.sourceforge.net/doc//misc/imagescgray.php

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

% Author: Nathanael Perraudin
% Date  : 14.03.2013


if nargin>1
    if nfig==0
        figure();
    else
        figure(nfig);
    end
end
if nargin>3
    subplot(subplot)
end
imagesc(im);
colormap gray;
hold on;
if nargin>2
    title(tit);
end
axis off;          
axis image;
drawnow;
end


