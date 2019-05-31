function varargout= mysvds(varargin)
% dbstop if error 
Y = varargin{1};
maxrank = varargin{2};
SzY = size(Y);
mz = min(SzY); Mz = max(SzY);

if nargout == 1
    if (Mz/mz>2.5) && (Mz >= 1e3)
        if SzY(1)>SzY(2)
            Cy = full(Y'*Y);my = max(Cy(:));
            if any(Cy(:))
                S = eigs(Cy/my,varargin{2:end});
                S = sqrt(diag(S));
            else
                S = zeros(maxrank,1);
            end
        else
            Cy = full(Y*Y');my = max(Cy(:));
            if any(Cy(:))
                S = eigs(Cy/my,varargin{2:end});
                S = sqrt(diag(S));
            else
                S = zeros(maxrank,1);
            end
        end
    else
        S = svds(Y,maxrank);
    end
    
    varargout{1} = S;
else
    if (Mz/mz>2.5) && (Mz >= 1e3)
        if SzY(1)>SzY(2)
            Cy = full(Y'*Y);my = max(Cy(:));
            if any(Cy(:))
                [V,S] = eigs(Cy/my,varargin{2:end});
                S = sqrt(diag(S));
                anix = ~isnan(S);V = V(:,anix); S = S(anix);
                
                U = Y*V*diag(1./S);
                S = diag(S);
                
                [Uq,Ur] = qr(U,0);
                [uu,S,vv] = svd(Ur*S);
                U = Uq*uu;
                V = V*vv;
            else
                U = zeros(SzY(1),maxrank);
                V = zeros(SzY(2),maxrank);
                S = zeros(maxrank);
            end
        else
            Cy = full(Y*Y');my = max(Cy(:));
            if any(Cy(:))
                [U,S] = eigs(Cy/my,varargin{2:end});
                S = sqrt(diag(S));
                anix = ~isnan(S);U = U(:,anix); S = S(anix);
                
                V = (U'*Y)'*diag(1./S)*1/sqrt(my);
                S = diag(S);
                
                [Vq,Vr] = qr(V,0);
                [vv,S,uu] = svd(Vr*S);
                U = U*uu;
                V = Vq*vv;
                
            else
                U = zeros(SzY(1),maxrank);
                V = zeros(SzY(2),maxrank);
                S = zeros(maxrank);
            end
        end
        
    else
        [U,S,V] = svds(Y,maxrank);
    end
    
    varargout{1} = U;
    varargout{2} = S;
    if nargout >=3
        varargout{3} = V;
    end
end
end