%%
function shiftfilter = shiftfilter_op(shiftfilter)
switch shiftfilter
    case 'mean'
        shiftfilter = @(x,dim) mean(x,dim);
    case 'median'
        shiftfilter = @(x,dim) median(x,dim);
    case 'bestapprox'
        shiftfilter = @(x,dim) bestaprox(x);
end
end
