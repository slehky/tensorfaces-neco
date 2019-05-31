
function [Swn,Sbn,weight] = sub_scattermatrices(Pn,label,Nsamples_per_class,scale_fl)
% Swn = Pn'*Sw*Pn
%     = Pn'*Pn - Pn'*B * diag(1./Nc)*B'*Pn
%     =
%
% Sbn = Pn'*Sb*Pn
%     = Pn'*B*(diag(1./Nc) - 1/N * ones(Nclasses)) * B'*Pn
%     = Pn'*B * diag(1./Nc)*B'*Pn - 1/N * Pn'*1 * 1'*Pn
%
% Tn = B'*Pn yields a matrix of size  Nclasses x Jn
%   Tn(n,j): sum of entries Pn(k,j)  where k in the class n
%
% label :
%
if nargin<4
    scale_fl = false;
end
Nsamples = numel(label);
if nargin < 3
    Nsamples_per_class = grpstats(ones(Nsamples,1),label,{'sum'});
end
weight = 1;
if scale_fl == true
    weight = max(abs(Pn(:)));
    Pn = Pn/weight;
    weight = weight^2;
end
Q = Pn'*Pn;
PnB = grpstats(Pn,label,{'sum'}); % B'*P : Nclasses x J

% sub- With-in scatter matrix
S = PnB' * diag(1./Nsamples_per_class) * PnB;
Swn = Q - S;
b = sum(PnB,1);
% sub- between scatter matrix
Sbn = S - 1/Nsamples*(b'*b);
end