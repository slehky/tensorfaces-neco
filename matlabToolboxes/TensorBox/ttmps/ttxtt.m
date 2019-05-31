function Z = ttxtt(Xtt,Ytt,mode,side)
% TT-tensor time tensor Y along modes
%%
% TENSORBOX, 2018
N = ndims(Ytt);

SzY = size(Ytt);
SzX = size(Xtt);
rankX = rank(Xtt);
rankY = rank(Ytt);

switch side
    case 'left'
        modes = 1:min(mode)-1;
        
        
        for n = modes
            if n == 1
                Z = squeeze(Xtt{n})' * squeeze(Ytt{n}); % R2 x S2
            else
                Z = Z * reshape(Ytt{n},rankY(n),[]);    % S2 x I2S3
                Z = reshape(Z,rankX(n)*SzY(n),[]);      % R2I2 x S3
                Z = reshape(Xtt{n},rankX(n)*SzY(n),[])' * Z; % R3 x S3
            end
        end
        Z = reshape(Z,[rankX(n+1), rankY(n+1)]);
        
        
    case 'right'
        modes = N:-1:max(mode)+1;
        
        
        for n = modes
            if n == N
                Z = Xtt{n}*Ytt{n}';                     % RN x SN
            else
                Z = Z * reshape(Ytt{n},[],rankY(n+1))';      % RX x I(n)S(n)
                Z = reshape(Xtt{n},rankX(n),[]) * reshape(Z,[],rankY(n)); % Rn x Sn
            end
        end
        
        Z = reshape(Z,[rankX(n), rankY(n)]);
        
    case 'both'
        
        Zright = ttxtt(Xtt,Ytt,mode,'right');
        Zleft = ttxtt(Xtt,Ytt,mode,'left');
        Z = kron(Zright,Zleft);
end


