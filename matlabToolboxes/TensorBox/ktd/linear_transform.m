function Patch_Opts = linear_transform(Patch_Opts)
% Prepare linear transformation operator for patterns in the KTD
% Name of transformation is specified by Patch_Opts.Transform.
%
% The forward transform operator is assigned to Patch_Opts.TF
% whereas inverse transform operator is given in Patch_Opts.iTF
%
%
% Phan Anh Huy, 2015

No_Groups = numel(Patch_Opts);

for k = 1:No_Groups
    Patch_curr = Patch_Opts(k);
    patch_size = Patch_curr.Size;
    
    if numel(patch_size) == 2
        switch Patch_curr.Transform
            case 'none'
                Sparse_tf = @(x) x ;
                IvSparse_tf = @(x) x;
                
            case 'dct'
                if prod(patch_size{end}) < 2000
                    LRmtx = 1;
                    for kn = 1:numel(patch_size{end})
                        if patch_size{end}(kn)>=3
                            mtx = dctmtx(patch_size{end}(kn));
                        else
                            mtx = speye(patch_size{end}(kn));
                        end
                        LRmtx = kron(mtx,LRmtx);
                    end
                    
                    Sparse_tf = @(x) x*LRmtx';
                    IvSparse_tf = @(x) x*LRmtx;
                    
                else
                    LRmtx = cell(numel(patch_size{end}),1);
                    for kn = 1:numel(patch_size{end})
                        if patch_size{end}(kn)>=3
                            mtx = dctmtx(patch_size{end}(kn));
                        else
                            mtx = eye(patch_size{end}(kn));
                        end
                        LRmtx{kn} = mtx;
                    end
                    Sparse_tf = @(x) double(tenmat(ttm(tensor(x,[size(x,1) patch_size{end}]),LRmtx,2:numel(patch_size{end})+1,'t'),1));
                    IvSparse_tf = @(x) double(tenmat(ttm(tensor(x,[size(x,1) patch_size{end}]),LRmtx,2:numel(patch_size{end})+1),1));
                     
                end
                
            case 'walsh' % or Walsh wavelet
                
                if prod(patch_size{end})<2000
                    LRmtx = 1;
                    for kn = 1:numel(patch_size{end})
                        if patch_size{end}(kn)>3
                            mtx = fwht(eye(patch_size{end}(kn)))*sqrt(patch_size{end}(kn));
                        else
                            mtx = speye(patch_size{end}(kn));
                        end
                        
                        LRmtx = kron(mtx,LRmtx);
                    end
                    
                    Sparse_tf = @(x) x*LRmtx';
                    IvSparse_tf = @(x) x*LRmtx;
                     
                else
                    LRmtx = cell(numel(patch_size{end}),1);
                    for kn = 1:numel(patch_size{end})
                        if patch_size{end}(kn)>3
                            mtx = fwht(eye(patch_size{end}(kn)))*sqrt(patch_size{end}(kn));
                        else
                            mtx = eye(patch_size{end}(kn));
                        end
                        
                        LRmtx{kn} = mtx;
                    end
                    
                    Sparse_tf = @(x) double(tenmat(ttm(tensor(x,[size(x,1) patch_size{end}]),LRmtx,2:numel(patch_size{end})+1,'t'),1));
                    IvSparse_tf = @(x) double(tenmat(ttm(tensor(x,[size(x,1) patch_size{end}]),LRmtx,2:numel(patch_size{end})+1),1));
                end
                
            otherwise % other wavelets
                
                try
                    wname = Patch_curr.Transform;
                    %wname = 'sym2';
                    %wname = 'coif2';
                    % Fast orthogonal wavelet transform with small patches
                    [Lo_D,Hi_D] = wfilters(wname);
                    Lx_Lmtx = convmtx(Lo_D(:),patch_size{end}(1)); %Lx_Lmtx = Lx_Lmtx(numel(Lo_D)/2+2:numel(Lo_D)/2+Ix(2,1)+1,:);
                    Hx_Lmtx = convmtx(Hi_D(:),patch_size{end}(1)); %Hx_Lmtx = Hx_Lmtx(numel(Lo_D)/2+2:numel(Lo_D)/2+Ix(2,1)+1,:);
                    
                    Lx_Rmtx = convmtx(Lo_D(:),patch_size{end}(2)); %Lx_Rmtx = Lx_Rmtx(numel(Lo_D)/2+2:numel(Lo_D)/2+Ix(2,1)+1,:);
                    Hx_Rmtx = convmtx(Hi_D(:),patch_size{end}(2)); %Hx_Rmtx = Hx_Rmtx(numel(Lo_D)/2+2:numel(Lo_D)/2+Ix(2,1)+1,:);
                    
                    Lmtx = [Lx_Lmtx ;Hx_Lmtx];
                    Rmtx = [Lx_Rmtx ;Hx_Rmtx];
                    
                    LRmtx = kron(Rmtx,Lmtx);
                    LRmtx = LRmtx/norm(LRmtx(:,1));
                    
                    if numel(patch_size{end}) > 2 && prod(patch_size{end}(3:end)) > 1
                        LRmtx = kron(speye(prod(patch_size{end}(3:end))),LRmtx);
                    end
                    
                    Sparse_tf = @(x) x*LRmtx';
                    IvSparse_tf = @(x) x*LRmtx;
                    
                catch
                    
                    Sparse_tf = @(x) x ;
                    IvSparse_tf = @(x) x;
                    
                end
        end
    else
        Sparse_tf = @(x) x ;
        IvSparse_tf = @(x) x;
    end
    
    Patch_curr.TF = Sparse_tf;
    Patch_curr.iTF = IvSparse_tf;
    
    Patch_Opts(k) = Patch_curr;
end
end