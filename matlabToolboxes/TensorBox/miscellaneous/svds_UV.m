function [Ux,Sx,Vx] = svds_UV(U,V,R)

try 
    [Ux,Sx,Vx] = svds(@(x,tflag) Afun(x,tflag),[size(U,1) size(V,1)],R);
catch 
    [Ux,Sx,Vx] = fsvds(@(x,tflag) Afun(x,tflag),[size(U,1) size(V,1)],R);
end

    function f = Afun(X,tfflag)
        switch tfflag
            case 'notransp'
                f = U*(V'*X);
            case 'transp'
                f = V*(U'*X);
        end
    end

end
