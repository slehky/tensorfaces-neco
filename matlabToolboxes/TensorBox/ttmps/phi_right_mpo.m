function phi_ge = phi_right_mpo(U,X,n,dir,phi_ge,tmode)
% Compute Phi_right = phi_>=n
% between a TT-matrix, A, and a TT-vector X
%
% The n-th core of the TT-matrix A  are diagonal-2 tensor of size
% K x 1 x I x K constructed from U{n}
%          A{n}(:,1,i,:) = diag(U{n}(:,k))
%
% - n:   the specific tensor mode or all if empty
%
% - 'dir' :  L2R or R2L
%          If R2L (right to left), Phi_right(n) is progressively computed
%          from Phi_right(n+1)
%          Otherwise it needs an entire computation.
% - tmode: 't' transpose or not
%
% Phan Anh Huy

N = numel(U);
if nargin<6
    tmode = 'n';  % normal X x U otherwise X x U^T
end
if nargin<5
    phi_ge = cell(N,1);
end
if nargin<4 || isempty(dir)
    dir = 'R2L';
end
if nargin<3 || isempty(n)
    n = 1;
end

if isa(X,'TTeMPS') || isa(X,'tt_tensor')
    switch dir
        case 'R2L'
            if n<=N
                % check phi_ge(n-1) is computed
                if n == N
                    phi_ge{n} = squeeze(X{n})*U{n};
                elseif isempty(phi_ge{n+1})
                    for k = N:-1:n
                        if k == N
                            phi_ge{k} = squeeze(X{k})*U{k};
                        else
                            phi_ge{k} = reshape(X{k},size(X{k},1),[])*khatrirao(phi_ge{k+1},U{k});
                        end
                    end
                else
                    phi_ge{n} = reshape(X{n},size(X{n},1),[])*khatrirao(phi_ge{n+1},U{n});
                end
            end
            
        case 'L2R'
            % Phi_right = phi_>=n :  R(n-1) x K
            for k = N:-1:n
                if k == N
                    phi_ge{k} = squeeze(X{k})*U{k};
                else
                    phi_ge{k} = reshape(X{k},size(X{k},1),[])*khatrirao(phi_ge{k+1},U{k});
                end
            end
            
    end
    
elseif  isa(X,'TTeMPO') ||  isa(X,'TTeMPS_op') || isa(X,'tt_matrix')
    % Only compute phi_right upto block_mode+1
    block_mode = find(cellfun(@(x) size(x,3),X.U,'uni',1) >1);
    n = max(n,block_mode+1);
    
    switch dir
        case 'R2L'
            % check phi_ge(n-1) is computed
            if n<=N
                if n == N
                    phi_ge{n} = squeeze(X{n})*U{n};
                elseif isempty(phi_ge{n+1})
                    for k = N:-1:n
                        if k == N
                            phi_ge{k} = squeeze(X{k})*U{k};
                        else
                            phi_ge{k} = reshape(X{k},size(X{k},1),[])*khatrirao(phi_ge{k+1},U{k});
                        end
                    end
                    
                    for k = N:-1:n
                        if k == N
                            phi_ge{k} = squeeze(X{k})*U{k};
                        else
                            phi_ge{k} = reshape(X{k},size(X{k},1),[])*khatrirao(phi_ge{k+1},U{k});
                        end
                    end
                else
                    phi_ge{n} = reshape(X{n},size(X{n},1),[])*khatrirao(phi_ge{n+1},U{n});
                end
            end
            
        case 'L2R'
            % Phi_right = phi_>=n :  R(n-1) x K
            for k = N:-1:n
                if k == N
                    phi_ge{k} = squeeze(X{k})*U{k};
                else
                    phi_ge{k} = reshape(X{k},size(X{k},1),[])*khatrirao(phi_ge{k+1},U{k});
                end
            end
            
    end
    
end
