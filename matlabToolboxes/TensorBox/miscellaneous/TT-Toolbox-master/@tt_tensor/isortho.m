function [state,err] = isortho(X,pos,side)
% Check if a core X{pos} is orthogonal from left or right side
%
% Phan Anh Huy, 2016
%
switch side
    case 'left'
        %Xn = reshape(X.U{pos},[],size(X.U{pos},3));
        Xn = reshape(X.core(X.ps(pos):X.ps(pos+1)-1),[],X.r(pos+1));
        F = Xn'*Xn;
    case 'right'
        %Xn = reshape(X.U{pos},size(X.U{pos},1),[]);
        Xn = reshape(X.core(X.ps(pos):X.ps(pos+1)-1),X.r(pos),[]);
        F = Xn*Xn';
end

err = norm(F - eye(size(F)),'fro');
state = err/numel(err) < 1e-6;