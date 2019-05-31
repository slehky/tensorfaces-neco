function Yh = kron_ishift(Yh,SzY,step)
%% Shift the data Yh back step size to its size

Yh = reshape(Yh,prod(SzY),[]);
for kshift = 1:size(Yh,2)
    shift_ix = ind2sub_full(2*step+1,kshift);
    shift_ix = shift_ix - step - 1;
    Yks = reshape(Yh(:,kshift),SzY);
    Yks = circshift(Yks,-shift_ix);
    Yh(:,kshift) = Yks(:);
end
%Yh = reshape(Yh,[],Noshifts]);