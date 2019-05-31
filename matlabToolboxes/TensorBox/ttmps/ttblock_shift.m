function X = ttblock_shift(X,nshift)
% X is a block TT-tensor
% i.e., a TT-matrix of size I1I2...IN x J1J2...JN
% where (N-1) Jm are 1s, and only one Jn ~= 1,
%
% nshift : +n to the right n modes
%          -n to the left n modes
% Phan Anh Huy, 2017
%

szR = cellfun(@(x) size(x,3),X.U,'uni',1);
%szL = cellfun(@(x) size(x,2),X.U,'uni',1);
block_mode = find(szR>1);
N = numel(szR);

if nshift>0
    % to right
    
    % Merge two cores
    for nct = 1:nshift
        if block_mode < N
            X1 = X.U{block_mode}; % Rn In K R(n+1)
            X2 = X.U{block_mode+1}; % R(n+1) I(n+1) R(n+2)
            szX1 = size(X1);
            szX2 = size(X2);
            if numel(szX1) < 4
                szX2(end+1:4) = 1;
            end
            if numel(szX2) <4
                szX2(end+1:4) = 1;
            end
            
            A = tt_prod(X1,X2);
            A = reshape(A,szX1(1)*szX1(2),[]);
            [QQ,RR] = qr(A,0); % RnIn x S(n+1)        S(n+1) * KI(n+1)R(n+2)
            
            % check and truncate the rank
            ss = sum(abs(RR).^2,2);
            eps_ = 1e-8;
            ixs = ss > eps_*sum(ss);
            R = sum(ixs); %  % rank may change
            X.U{block_mode} = reshape(QQ(:,ixs),szX1(1),szX1(2),1,[]);
            X.U{block_mode+1} = reshape(RR(ixs,:),R,szX1(3),szX2(2),[]);
            X.U{block_mode+1} = permute(X.U{block_mode+1},[1 3 2 4]);
            block_mode = block_mode+1;
        end
        
    end
else % to left
    
    for nc = 1:abs(nshift)
        if block_mode > 1
            % Merge two cores
            X1 = X.U{block_mode-1}; % R(n-1) I(n-1) R(n)
            X2 = X.U{block_mode};   % Rn In K R(n+1)
            X2 = permute(X2,[1 3 2 4]); % Rn K In R(n+1)
            szX1 = size(X1);
            szX2 = size(X2);
            if numel(szX1) < 4
                szX2(end+1:4) = 1;
            end
            if numel(szX2) <4
                szX2(end+1:4) = 1;
            end
            
            A = tt_prod(X1,X2);
            
            A = reshape(A,[],szX2(3)*szX2(4));
            [QQ,RR] = qr(A',0); % RnIn x S(n+1)        S(n+1) * KI(n+1)R(n+2)
            
            % check and truncate the rank
            ss = sum(abs(RR).^2,2);
            eps_ = 1e-8;
            ixs = ss > eps_*sum(ss);
            R = sum(ixs); %  % rank may change
            X.U{block_mode} = reshape(QQ(:,ixs)',R,szX2(3),1,[]);
            X.U{block_mode-1} = reshape(RR(ixs,:)',szX1(1),szX1(2),[],R);
            block_mode = block_mode-1;
        end
    end
end
