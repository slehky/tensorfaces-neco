function colorizedstring = colorizestring(color, stringtocolorize)
%colorizestring wraps the given string in html that colors that text.
    colorizedstring = ['<font color="', color, '">', stringtocolorize, '</font>'];
end