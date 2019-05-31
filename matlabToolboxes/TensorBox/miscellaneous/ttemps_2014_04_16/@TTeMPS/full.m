function y = full( x )
    %FULL Convert TTeMPS tensor to full array
    %   X = FULL(X) converts the TTeMPS tensor X to a (X.order)-dimensional full array of size
    %   X.size(1) x X.size(2) x ... x X.size(X.order)
    %
    %	Use with care! Result can easily exceed available memory.
    %
    %   See also SUBSREF.

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2014
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt


    y = x.U{1};
    y = reshape( y, x.rank(1)*x.size(1),[]);
    for i = 2:x.order
        U_temp = unfold( x.U{i}, 'right');
        y = y * U_temp;
        y = reshape( y,[],x.rank(i+1));
    end
    y = reshape( y, [x.rank(1) x.size x.rank(end)]); % fixed when the 
    y = squeeze(y); % The first and the last dimensions might show the tail ranks
end
