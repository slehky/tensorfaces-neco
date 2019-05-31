function phi_le = phi_left(U,X,n,dir,phi_le)
% Compute Phi_left = phi_<=n
% between a TT-matrix, A, and a TT-vector X
%
% The n-th core of the TT-matrix A  are diagonal-2 tensor of size
% K x 1 x I x K constructed from U{n}
%          A{n}(:,1,i,:) = diag(U{n}(:,k))
%
% - n:   the specific tensor mode or all if empty
%
% - 'dir' :  L2R or R2L
%          If L2R (left to right), Phi_left(n) is progressively computed
%          from Phi_left(n-1)
%          Otherwise needs an entire computation.
%
% Phan Anh Huy

N = numel(U);
if nargin<5
    phi_le = cell(N,1);
end
if nargin<4 || isempty(dir)
    dir = 'L2R';
end
if nargin<3 || isempty(n)
    n = N;
end

if isa(X,'TTeMPS') || isa(X,'tt_tensor')
    
    switch dir
        case 'L2R'
            % check phi_le(n-1) is computed
            if n == 1
                phi_le{n} = U{n}'*reshape(X{n},size(X{n},2),[]);
            elseif isempty(phi_le{n-1})
                for k = 1:n
                    if k == 1
                        %phi_le{k} = U{k}'*squeeze(X{k});
                        phi_le{k} = U{k}'*reshape(X{k},size(X{k},2),[]);
                    else
                        phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                    end
                end
            else
                phi_le{n} = khatrirao(U{n},phi_le{n-1}')'*reshape(X{n},size(X{n},1)*size(X{n},2),[]);
            end
            
        case 'R2L'
            for k = 1:n
                if k == 1
                    phi_le{k} = U{k}'*reshape(X{k},size(X{k},2),[]);
                else
                    phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                end
            end
            
    end
    
elseif  isa(X,'TTeMPO') ||  isa(X,'TTeMPS_op') || isa(X,'tt_matrix')
    % Only compute phi_left upto block_mode-1
    block_mode = find(cellfun(@(x) size(x,3),X.U,'uni',1) >1);
    n = min(n,block_mode-1);
    
    switch dir
        case 'L2R'
            % check phi_le(n-1) is computed
            
            if n == 1
                phi_le{n} = U{n}'*reshape(X{n},size(X{n},1)*size(X{n},2),[]);
            elseif isempty(phi_le{n-1})
                for k = 1:n
                    if k == 1
                        phi_le{k} = U{k}'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                    else
                        phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                    end
                end
            else
                phi_le{n} = khatrirao(U{n},phi_le{n-1}')'*reshape(X{n},size(X{n},1)*size(X{n},2),[]);
            end
            
        case 'R2L'
            for k = 1:n
                if k == 1
                    phi_le{k} = U{k}'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                else
                    phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X{k},size(X{k},1)*size(X{k},2),[]);
                end
            end
            
    end
end
