function [foldingrule,cfold,foldcrib] = unfoldingstrategy(U,mindim,cribbased,sigma)
% This function suggests the tensor unfolding for an order-N tensor in the
% Krukal form with factor matrices U{1}, U{2}, ..., U{N} such that the loss
% of accuracy when using the FCP algorithm is as little as possible.
%
% Input:
%  U:   Factor matrices
%  dim: order of the unfolded tensor (e.g., 3)
%
% Output:
%  P:  ktensor of estimated factors
%  output:
%
% REF:
%
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012  
%
%
% [2] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, "Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition",
% http://arxiv.org/abs/1209.3215, 2012. 
%
% [3] A. H. Phan, P. Tichavsky, and A. Cichocki, "On fast computation of
% gradients for CP algorithms", http://arxiv.org/abs/1204.1586, 2012,
% submitted.
%
%
% Example: 
%     c = [.1 .99 .1 .7   .99 .7 .8];    % correlation coefficients
%     N = numel(c);
%     U = cell(N,1);
%     for n = 1:N
%         U{n} = gen_matrix(R,R,c(n));
%     end
%     % Find the optimal tensor unfolding rule
%     [foldingrule,cfold] = unfoldingstrategy(U,3);
%     
%
% See also: cp_fcp.
%
% This algorithm is a part of the TENSORBOX, 2012.
%
% Copyright Phan Anh Huy, 04/2012
%
%
if (nargin<2) || isempty(mindim)
    mindim = 3;
end
if (nargin<3) || isempty(cribbased)
    cribbased = false;
end

N = numel(U);R = size(U{1},2);I = cellfun(@(x) size(x,1),U);
% C = zeros(R,R,N);c = zeros(1,N);
% for n = 1:N
%     C(:,:,n) = U{n}'*U{n}; % correlation matrices
%     c(n) = sum(sum(abs(triu(C(:,:,n),1))))/(R*(R-1)/2);
% end
% [c,modeind] = sort(c,'ascend');
% foldingrule = num2cell(modeind);
% cfold = c;
% Cfold = C(:,:,modeind);
% 
% while numel(cfold)>dim
%     % Fold the last two modes
%     foldingrule{end-1} = sort([foldingrule{end-1} foldingrule{end}]);
%     Cfold(:,:,end-1) = Cfold(:,:,end-1).*Cfold(:,:,end);
%     Cfold(:,:,end) = [];
%     cfold(end-1) = sum(sum(abs(triu(Cfold(:,:,end),1))))/(R*(R-1)/2);
%     cfold(end) = [];
%     
%     [cfold,modeind] = sort(cfold,'ascend');
%     Cfold = C(:,:,modeind);
%     foldingrule = foldingrule(modeind);
% end

C = zeros(R,R,N);
C0 = zeros(R,R,N);
for n = 1:N
    C0(:,:,n) = U{n}'*U{n};
    
    u = bsxfun(@rdivide,U{n},sqrt(sum(U{n}.^2)));
    C(:,:,n) = u'*u;
end

Corr_deg = reshape(C,[],N);
Corr_deg(1:R+1:end,:) = [];
Corr_deg = abs(Corr_deg);

% [min(Corr_deg); max(Corr_deg)]-cval
Corr_deg0 = Corr_deg;
 
corr_bin = cos((90:-1:0)/180*pi);

Corr_deg = Corr_deg0;
foldingrule = num2cell(1:N);

if cribbased == 0
    % aux0 = hist(Corr_deg,corr_bin);
    
    while numel(foldingrule) > mindim
        %%
        Ncur = size(Corr_deg,2);
        newCorr_deg = khatrirao(Corr_deg',Corr_deg')';
        newCorr_deg(:,1:Ncur+1:end) = nan;
        aux = hist(newCorr_deg,corr_bin);
        lstbin = find(sum(aux>0,2),1,'last');
        hcdf = sum(aux(lstbin:end,:));
        [hcdf,hcorr_mode] = sort(hcdf,'ascend');
        
        ind = find(hcdf == hcdf(end));
        sel_mode = hcorr_mode(ind);
        [mode1,mode2] = ind2sub([Ncur Ncur],sel_mode);
        sel_mode = sort([mode1; mode2])';
        sel_mode = unique(sel_mode,'rows');
        
        % If sel_mode has more than one row: there are many possible
        % combinations, check for higher bins
        if size(sel_mode,1)>1
            sel_mode_lin = sub2ind([Ncur Ncur],sel_mode(:,1),sel_mode(:,2));
            ind = 1:numel(sel_mode_lin);
            while lstbin >1
                lstbin = lstbin - 1;
                hcdf = sum(aux(lstbin:end,sel_mode_lin));
                [hcdf,hcorr_mode] = sort(hcdf,'ascend');
                ind = find(hcdf == hcdf(end));
                ind = hcorr_mode(ind);
                if numel(ind) == 1
                    break
                end
            end
            
            sel_mode = sel_mode(ind,:);
            sel_mode_lin = sel_mode_lin(ind);
            
            if numel(sel_mode_lin) > 1 % there are still many combinations, try one with less permutation
                %newfolding = cell(numel(sel_mode_lin),1);
                complexoffolding = zeros(numel(sel_mode_lin),1);
                for kss = 1:numel(sel_mode_lin)
                    newfolding = sort([foldingrule{sel_mode(kss,1)} foldingrule{sel_mode(kss,2)}]);
                    % complexity of foldings is differnces between combination modes
                    complexoffolding(kss) = sum(diff(newfolding)-1);
                end
                % select foldings with the lowest complexity of tensor permutation
                [foe,imin] = min(complexoffolding);
                sel_mode = sel_mode(imin,:);
            end
        end
        
        foldingrule{sel_mode(1)} = sort([foldingrule{sel_mode(1)} foldingrule{sel_mode(2)}]);
        foldingrule(sel_mode(2))= [];
        Corr_deg(:,sel_mode(1)) = Corr_deg(:,sel_mode(1)) .* Corr_deg(:,sel_mode(2));
        Corr_deg(:,sel_mode(2)) = [];
        %     fprintf('Suggested unfolding rule  %s\n',foldingrule2char(foldingrule));
    end
    
    cfold = mean(Corr_deg);
    
elseif cribbased == 1
    Cfold = C0; 
    Corr_degfold = Corr_deg;
    Ifold = I;
    while numel(foldingrule) > mindim
        Ncur = numel(foldingrule);
        
        aux = mean(Corr_degfold);
        [foe,sel_mode(1)] = max(aux);
        
        modes = setdiff(1:Ncur,sel_mode(1));
        Foldcrib = zeros(numel(modes),(Ncur-1)*R);

        for km = 1:numel(modes)
            nmode = modes(km);
            
            %% Compute CRIB
            newfoldingrule = foldingrule;
            newfoldingrule{sel_mode(1)} = sort([newfoldingrule{sel_mode(1)} newfoldingrule{nmode}]);
            newfoldingrule(nmode)= [];
            
            newCfold = Cfold;
            newCfold(:,:,sel_mode(1)) = newCfold(:,:,sel_mode(1)) .* newCfold(:,:,nmode);
            newCfold(:,:,nmode)  = [];
            
            newIfold = Ifold;
            newIfold(sel_mode(1)) = newIfold(sel_mode(1)) * newIfold(nmode);
            newIfold(nmode) = [];
            
            Nfold = numel(newfoldingrule);
            
            
            newfoldcrib = zeros(Nfold,R);
            for n = 1:Nfold
                newCfoldn = newCfold(:,:,[n 1:n-1 n+1:Nfold]);
                for r = 1:R
                    newCfoldn_r = newCfoldn([r 1:r-1 r+1:end],:,:);
                    newCfoldn_r = newCfoldn_r(:,[r 1:r-1 r+1:end],:);
                    %newfoldcrib(n,r) = fastcribCP2(newCfoldn_r,newIfold(n)) ;
                    newfoldcrib(n,r) = cribNb(newCfoldn_r,newIfold(n)) ;
                end
            end
            
            Foldcrib(km,:) = newfoldcrib(:);
            %fprintf('Stage %d, newfoldCRIB %s \n',ka,sprintf('%2.2f dB  ',-10*log10(mean(newfoldcrib,2))))
        end
        [foe,temp] = min(mean(Foldcrib,2));
        sel_mode(2) = modes(temp);
        
        newfoldcrib = reshape(Foldcrib(temp,:),[],R);
        foldingrule{sel_mode(1)} = sort([foldingrule{sel_mode(1)} foldingrule{sel_mode(2)}]);
        foldingrule(sel_mode(2))= [];
        
        %Ncur = numel(foldingrule);
        
%         Ufold = cell(Nfold,1);
%         for kn = 1:Nfold
%             Ufold{kn} = khatrirao(U(foldingrule{kn}(end:-1:1)));
%         end
%         Cfold = zeros(R,R,Nfold);
%         for kn = 1:Nfold
%             Cfold(:,:,kn) = Ufold{kn}'*Ufold{kn};
%         end
%         Ifold = cellfun(@(x) size(x,1),Ufold);
%         
%         Corr_degfold = reshape(Cfold,R^2,[]);
%         Corr_degfold(1:R+1:end,:) = [];
%         Corr_degfold = abs(Corr_degfold);

        
        Cfold(:,:,sel_mode(1)) = Cfold(:,:,sel_mode(1)) .* Cfold(:,:,sel_mode(2));
        Cfold(:,:,sel_mode(2))  = [];
        
        Corr_degfold(:,sel_mode(1)) = Corr_degfold(:,sel_mode(1)) .* Corr_degfold(:,sel_mode(2));
        Corr_degfold(:,sel_mode(2)) = [];
        
        Ifold(sel_mode(1)) =  Ifold(sel_mode(1)) * Ifold(sel_mode(2));
        Ifold(sel_mode(2))= [];
        
        
        
    end
    cfold = mean(Corr_degfold);
    foldcrib = newfoldcrib;
    
elseif cribbased == 2 % fully search through all combinations 
    
    Cfold = C0;
    Ifold = I;
    Corr_degfold = Corr_deg;
    foldingrule = num2cell(1:N);
    while numel(foldingrule)>mindim
        
        Nfold0 = numel(foldingrule);
%         if Nfold0 == N-1 % Fold the two modes with highest correlation degree
%             aux = mean(Corr_degfold);
%             [aux,c_order] = sort(aux,'descend');
%             
%             % If there are many modes with the same highest correlation
%             % degree, select modes with sortest dimensions
%             find(aux == aux(1));
%             
%             
%             
%         else
         
        FoldCRIB = nan(Nfold0,Nfold0,(Nfold0-1)*R);
        for kfmode1 = 1:Nfold0-1
            for kfmode2 = kfmode1+1:Nfold0
                newfoldingrule = foldingrule;
                newfoldingrule{kfmode1} = [newfoldingrule{kfmode1} newfoldingrule{kfmode2}];
                newfoldingrule(kfmode2) = [];
                
                Ifoldnew = Ifold;
                Ifoldnew(kfmode1) = Ifoldnew(kfmode1)*Ifoldnew(kfmode2);
                Ifoldnew(kfmode2) = [];
                
                Cfoldnew = Cfold;
                Cfoldnew(:,:,kfmode1) = Cfoldnew(:,:,kfmode1).*Cfoldnew(:,:,kfmode2);
                Cfoldnew(:,:,kfmode2) = [];
                
                Reducerule = newfoldingrule;
                Nfold = numel(Reducerule);
                
                newfoldcrib = zeros(Nfold,R);
                for n = 1:Nfold
                    Cn = Cfoldnew(:,:,[n 1:n-1 n+1:end]);
                    for r = 1:R
                        Cnr = Cn([r 1:r-1 r+1:end],:,:);
                        Cnr = Cnr(:,[r 1:r-1 r+1:end],:);
                        newfoldcrib(n,r) = cribNb(Cnr,Ifoldnew(n)) ;
                    end
                end
                
                fprintf('newfoldCRIB %s, aver. %2.2f dB, Rule %s\n',...
                    sprintf('%2.2f dB  ',-10*log10(sigma^2*mean(newfoldcrib,2))),...
                    -10*log10(sigma^2*mean(newfoldcrib(:))),...
                    foldingrule2char(newfoldingrule))
                
                FoldCRIB(kfmode1,kfmode2,:) = newfoldcrib(:);
            end
        end
        maxcrib = max(FoldCRIB,[],3);
        [foe,minrc] = nanmin(maxcrib(:));
        [kfmode1,kfmode2] = find(maxcrib == foe);
        
        if numel(kfmode1)>1
            [foe,imin] = min(Ifold(kfmode1) .* Ifold(kfmode2));
            kfmode1 = kfmode1(imin);
            kfmode2 = kfmode2(imin);
        end
        
        foldingrule{kfmode1} = [foldingrule{kfmode1} foldingrule{kfmode2}];
        foldingrule(kfmode2) = [];
        
        Cfold(:,:,kfmode1) = Cfold(:,:,kfmode1).*Cfold(:,:,kfmode2);
        Cfold(:,:,kfmode2) = [];
        
        Corr_degfold(:,kfmode1) = Corr_degfold(:,kfmode1) .* Corr_degfold(:,kfmode2);
        Corr_degfold(:,kfmode2) = [];
        
        newfoldcrib = FoldCRIB(kfmode1,kfmode2,:);
        newfoldcrib =  reshape(newfoldcrib,[],R);
        
        Ifold(kfmode1) = Ifold(kfmode1) * Ifold(kfmode2);
        Ifold(kfmode2) = [];
        
        cprintf('*red','newfoldCRIB %s, aver. %2.2f dB, Rule %s\n',...
            sprintf('%2.2f dB  ',-10*log10(sigma^2*mean(newfoldcrib,2))),...
            -10*log10(sigma^2*mean(newfoldcrib(:))),...
            foldingrule2char(foldingrule))
    end
    cfold = mean(Corr_degfold);
    foldcrib = newfoldcrib;
end

% fprintf('Suggested unfolding rule  %s\n',foldingrule2char(foldingrule));
% aux = hist(Corr_deg,corr_bin);
% plot(corr_bin,aux)
% 
% for km = 1:N
% fprintf('Mode %d, Corr. [%.2f -%.2f], Aver. %.2f \n',km,...
%     min(Corr_deg0(:,km)),max(Corr_deg0(:,km)),mean(Corr_deg0(:,km)))
% end
% 
% Ncur = size(Corr_deg,2);
% for km = 1:Ncur
% fprintf('Mode %d, Corr. [%.2f -%.2f], Aver. %.2f \n',km,...
%     min(Corr_deg(:,km)),max(Corr_deg(:,km)),mean(Corr_deg(:,km)))
% end