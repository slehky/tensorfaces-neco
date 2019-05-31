function [Urc,patchlocs] = gen_patches(Yn,ir,ic,d,step_d,patch1st,w)
% 
% collect patchs of size (d x d) around block of size wxw whose topleft is (ir,ic)
% top-right of a patch is (ir-d,ic-d+k)
% Number of patches is 4*(d+1)
% patch1st : index of the first pactch
% Output are patches whose indices are  patch1st:step_d:4*(d+1).
% 
%
% Phan Anh Huy, 2017
if nargin < 7
    w = 1; 
end

if nargin < 6
    patch1st = 1;
end
% No_patches = 4*(d+w)*d;

patchlocs = zeros(1,2);
cnt = 0;
% Patch whose top-right is (ir-d,ic-d+n) % move along the first row
for n = 0:d+w
    cnt = cnt+1;
    patchlocs(cnt,:) = [ir-d, ic-d+n];
end

% Patch whose top-right is (ir-d+k,ic+w)  % move along the last column
for n = 1:d+w
    cnt = cnt+1;
    patchlocs(cnt,:) = [ir-d+n, ic+w];
end

% Patch whose top-right is (ir+w,ic+k-1)  % move along the last row (ir+w,
% ic+k-1)
for n = d+w-1:-1:0
    cnt = cnt+1;
    patchlocs(cnt,:) = [ir+w, ic-d+n];
end
% Patch whose top-right is (ir-d+k,ic-d) % move along the first column
% (ir-d+n,ic-1)
for n = d+w-1:-1:1
    cnt = cnt+1;
    patchlocs(cnt,:) = [ir-d+n, ic-d];
end

patchlocs = circshift(patchlocs,-patch1st+1,1);
patchlocs_sel = patchlocs(1:step_d:end,:);

Urc = zeros(d^2,size(patchlocs_sel,1));
for ks = 1:size(patchlocs_sel,1)
    irs = patchlocs_sel(ks,1);
    ics = patchlocs_sel(ks,2);
    temp = Yn(irs:irs+d-1,ics:ics+d-1);
    Urc(:,ks) = temp(:);
end

% Urc = zeros(d^2,1);
% for n = 0:step_d:d+1
%     cnt = cnt+1;
%     temp = Yn(ir-d:ir-1,ic-d+n:ic+n-1);
%     Urc(:,cnt) = temp(:);
%     patchlocs(cnt,:) = [ir-d, ic-d+n];
% end
% % Patch whose top-right is (ir-d+k,ic-d)
% for n = step_d:step_d:d+1
%     cnt = cnt+1;
%     temp = Yn(ir-d+n:ir+n-1,ic-d:ic-1);
%     Urc(:,cnt) = temp(:);
%     patchlocs(cnt,:) = [ir-d+n, ic-d];
% end
% % Patch whose top-right is (ir+1,ic+k-1)
% for n = step_d:step_d:d+1
%     cnt = cnt+1;
%     temp = Yn(ir+1:ir+d,ic-d+n:ic+n-1);
%     Urc(:,cnt) = temp(:);
%     patchlocs(cnt,:) = [ir+1, ic-d+n];
% end
% % Patch whose top-right is (ir-d+k,ic+1)
% for n = step_d:step_d:d+1-step_d
%     cnt = cnt+1;
%     temp = Yn(ir-d+n:ir+n-1,ic+1:ic+d);
%     Urc(:,cnt) = temp(:);
%     patchlocs(cnt,:) = [ir-d+n, ic+1];
% end
end