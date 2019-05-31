function [Patch_Opts,Patch_Sizes] = correct_patch_size(Data_Sz,Patch_Opts)
% Correct patch sizes 
% Data_Sz: size of data
No_Groups = numel(Patch_Opts); % SHOULD BE ONE for single approximation term
Patch_Sizes = {Patch_Opts.Size};
Data_ord = numel(Data_Sz);

for kp = 1:No_Groups
    pt_size = Patch_Sizes{kp};
    
    if ischar(pt_size) && strcmp(pt_size,'quantum')
        % Quantum KTD with patches of size 2x2x1x1x...x1
        pt_size = [];
        for kn = 1:numel(Data_Sz)
            temp = 2*ones(ceil(log2(Data_Sz(kn))),1);
            
            if prod(temp) == Data_Sz(kn)
                
                if isempty(pt_size)
                    pt_size = [pt_size temp];
                else
                    idx = find(prod(pt_size,2)<4,1,'first');
                    temp = [ones(idx-1,1);temp];
                    pt_size = [pt_size;ones(idx-1,size(pt_size,2))];
                    pt_size = [pt_size temp];
                end
            else
                if prod(pt_size(1,:))*Data_Sz(kn) < 30
                    pt_size = [pt_size [Data_Sz(kn) ; ones(size(pt_size,1)-1,1)]];
                else
                    pt_size = [pt_size ones(size(pt_size,1),1); ones(1,size(pt_size,2)) Data_Sz(kn)];
                end
            end
        end
        if size(pt_size,1)>3
            pt_size = [prod(pt_size(1:3,:),1); pt_size(4:end,:)];
        end
        pt_size = pt_size(end:-1:1,:);
        pt_size = mat2cell(pt_size,ones(size(pt_size,1),1),size(pt_size,2));
        pt_size = pt_size';
    else
        pt_size = cellfun(@(x) [x(1:min(numel(x),Data_ord)) ones(1,Data_ord-numel(x))],pt_size,'uni',0);
        pt_size{end} = Data_Sz./prod(cell2mat(reshape(pt_size(1:end-1),[],1)),1);
    end
    Patch_Sizes{kp} = pt_size;
    Patch_Opts(kp).Size = pt_size;
end
end
