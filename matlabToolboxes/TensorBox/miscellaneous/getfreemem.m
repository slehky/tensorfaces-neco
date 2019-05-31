function memfree = getfreemem
if ispc
    [user sys] = memory;
    memfree = sys.PhysicalMemory.Available;
end
if ismac || isunix
    [foe,memfree] = unix(sprintf('top -l 1 | grep PhysMem: | awk ''{print $10}'''));
    idx = strfind(memfree,'M');
    memfree = str2num(memfree(1:idx-1))*1e6;
end