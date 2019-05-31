function fstr = foldinrule2char(foldingrule)
% This function is a part of the TENSORBOX, 2012.
% Copyright Phan Anh Huy, 04/2012

if iscell(foldingrule)
    fstr = {};
    for kf = 1:numel(foldingrule)
        if numel(foldingrule{kf}) > 1
            str = ['('];
            for kk = 1:numel(foldingrule{kf})-1
                str = [str sprintf('%d,',foldingrule{kf}(kk))];
            end
            str = [str sprintf('%d)',foldingrule{kf}(end))];
            fstr{kf} = str;
        else
            fstr{kf} = sprintf('%d',foldingrule{kf});
        end
        if kf < numel(foldingrule)
            fstr{kf} = [fstr{kf} ', '];
        end
    end
    fstr = cell2mat(fstr);
end
fstr = ['[' fstr ']'];