function Yh = bestaprox(Yshift)
%% Find best approximation from estimatations with shift 
%  Used with the KTD with shift
% 
Q = Yshift'*Yshift;
[uu,ss] = svds(Q,1);
Yh = Yshift*uu;
Ymean = mean(Yshift,2);
Yh = Yh * mean(abs(Ymean))/mean(abs(Yh));
Yh = Yh * sign(mean(Yh))*sign(mean(Ymean));
end