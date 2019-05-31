function phi_le = phi_left(U,X_mps,n,dir,phi_le)
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
if nargin<5 || isempty(phi_le) 
    phi_le = cell(N,1);
end
if nargin<4 || isempty(dir)
    dir = 'R2L';
end
if nargin<3 || isempty(n)
    n = N;
end

switch dir
    case 'L2R'
        % check phi_le(n-1) is computed
        if n>=1
            if n == 1
                phi_le{n} = U{n}'*reshape(X_mps{n},size(X_mps{n},2),[]);
            elseif isempty(phi_le{n-1})
                for k = 1:n
                    if k == 1
                        %phi_le{k} = U{k}'*squeeze(X_mps{k});
                        phi_le{k} = U{k}'*reshape(X_mps{k},size(X_mps{k},2),[]);
                    else
                        phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X_mps{k},size(X_mps{k},1)*size(X_mps{k},2),[]);
                    end
                end
            else
                phi_le{n} = khatrirao(U{n},phi_le{n-1}')'*reshape(X_mps{n},size(X_mps{n},1)*size(X_mps{n},2),[]);
            end
        end
        
    case 'R2L'
        for k = 1:n
            if k == 1
                phi_le{k} = U{k}'*reshape(X_mps{k},size(X_mps{k},2),[]);
            else
                phi_le{k} = khatrirao(U{k},phi_le{k-1}')'*reshape(X_mps{k},size(X_mps{k},1)*size(X_mps{k},2),[]);
            end
        end
        
end

