function [Yshift,Noshifts] = kron_shift(Y,step)
%% Shift the data Yh along its dimensions step  
% THere are total prod(2*step+1) version shifted from the data.

SzY = size(Y);
Noshifts = prod(2*step+1);
Yshift = zeros(numel(Y),Noshifts);
for kshift = 1:Noshifts
    shift_ix = ind2sub_full(2*step+1,kshift);
    shift_ix = shift_ix - step - 1;
    Yshift(:,kshift) = reshape(circshift(Y,shift_ix),[],1);
end
Yshift = reshape(Yshift,[SzY Noshifts]);