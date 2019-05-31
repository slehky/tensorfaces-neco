function ret=LineCon(f1, f2, f3, fp1, fp2, fp3);
%  Checks whether the middle point is located below or on the line connecting its neighbours.
%
% INPUT
% 
% f1,  f2,  f3  : goodness-of-fit values for the three points
% fp1, fp2, fp3 : number of free parameters for the three points
%
% OUTPUT
%
%  0 : The middle point is located below or on the line connecting its neighbours.
%  1 : The middle point is not located below or on the line connecting its neighbours.
%
% Written by Urbano Lorenzo-Seva, Rovira i Virgili University (Last update: October 25, 2007)
%

Ct = ((fp2-fp1)*(f3-f1) )/(fp3-fp1);
ft = f1+Ct;
if ft>=f2,
    ret=0;
else
    ret=1;
end;

return;