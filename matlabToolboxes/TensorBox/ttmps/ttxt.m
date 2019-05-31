function Z = ttxt(Xtt,Y,mode,side)
% TT-tensor Xtt time tensor Y along modes of Y to the left or right to
% "mode"
%
% Phan Anh Huy

N = ndims(Y);

SzY = size(Y);
% SzX = size(Xtt);
rankX = rank(Xtt);


switch side
    case 'left'
        modes = 1:min(mode)-1;

        if ~isempty(modes)
            
            for n = modes
                if n == 1
                    if isa(Y,'sptensor') % update on Feb 2nd
                        %Z = sptenmat(Y,1);
                        %Z = sparse(Z);
                        Z = reshape(Y,[SzY(1),prod(SzY(2:end))]);
                        Z = ttm(Z,reshape(Xtt{n},SzY(1),[])',1); % R2 x (I2 ... IN)
                    else
                        Z = reshape(Y,[SzY(1),prod(SzY(2:end))]);
                        Z = reshape(Xtt{n},SzY(1),[])' * Z; % R2 x (I2 ... IN)
                    end
                    
                else
                    if isa(Y,'sptensor') % update on Feb 2nd
                        %Z = sptenmat(Y,1);
                        %Z = sparse(Z);
                        Z = reshape(Z,[rankX(n)*SzY(n),prod(SzY(n+1:end))]);
                        Z = ttm(Z,reshape(Xtt{n},rankX(n)*SzY(n),[])',1); % R2 x (I2 ... IN)
                    else
                        
                        Z = reshape(Z,rankX(n)*SzY(n),[]); % In x (In+1... IN) R2 R3 ...R(n-1)
                        Z = reshape(Xtt{n},rankX(n)*SzY(n),[])' * Z; % R2 x (I2 ... IN)
                    end
                end
            end
            
            Z = reshape(Z,[rankX(n+1), SzY(n+1:end)]);
            
        else
            Z = Y;
        end
        
    case 'right' % contract (N-n+1) cores
        modes = N:-1:max(mode)+1;
        
        if ~isempty(modes)
            modes_X = ndims(Xtt):-1:ndims(Xtt)-numel(modes)+1;
            for n = 1:numel(modes)
                nX = modes_X(n);
                nY = modes(n);
                
                if nY == N
                    Z = reshape(Y,[],SzY(N));
                    
                    if ismatrix(Xtt{nX})
                        Z = Z * Xtt{nX}'; % R2 x (I2 ... IN)
                    else % Xn : rn x In x R
                        % When Xtt is a block TT, the last core is not a
                        % matrix but an order-3 tensor
                        
                        Z = Z * reshape(permute(Xtt{nX},[2 1 3]),size(Xtt{nX},2),[]); % R2 x (I2 ... IN)
                        Z = reshape(Z,[],size(Xtt{nX},3))'; % R x I1 x ... In-1 x Rn
                    end
                else
                    Z = reshape(Z,[],rankX(nX+1)*SzY(nY)); % In x (In+1... IN) R2 R3 ...R(n-1)
                    Z = Z *  reshape(Xtt{nX},[],rankX(nX+1)*SzY(nY))'; % R2 x (I2 ... IN)
                end
            end
            
            Z = reshape(Z,[size(Xtt{ndims(Xtt)},3) SzY(1:nY-1) rankX(nX)]);
            Z = permute(Z,[2:ndims(Z) 1]);
        else
            Z = Y;
        end
        
    case 'both'
        
        Z = ttxt(Xtt,Y,mode,'right');
        Z = ttxt(Xtt,Z,mode,'left');
        
%     case 'between' % between two modes
%         
%         modes = mode(1)+1:mode(2)-1;
%         if ~isempty(modes)
%             for n = modes
%                 if n == N
%                     Z = reshape(Y,[],SzY(N));
%                     Z = Z * Xtt{n}'; % R2 x (I2 ... IN)
%                 else
%                     Z = reshape(Z,[],rankX(n+1)*SzY(n)); % In x (In+1... IN) R2 R3 ...R(n-1)
%                     Z = Z *  reshape(Xtt{n},[],rankX(n+1)*SzY(n))'; % R2 x (I2 ... IN)
%                 end
%             end
%             
%             Z = reshape(Z,[SzY(1:n-1) rankX(n)]);
%         else
%             Z = Y;
%         end
end


