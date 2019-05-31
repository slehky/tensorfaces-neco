function [A,G] = ntd_initialize(Y,init,orthoforce,R)
% Initialization for NTD algorithms
% Output:   factors A and core tensor G
% Copyright by Anh Huy Phan
% Ver 1.0 12/2010, Anh Huy Phan
% Please do not distribute or use them in your publications or 
% your softwares without written agreement by Anh Huy Phan.
%
% This file must not be included in comercial
% software distributions without written agreement by Anh Huy Phan.

N = ndims(Y);In = size(Y);
if iscell(init)
    if numel(init) ~= N+1
        error('OPTS.init does not have %d cells',N+1);
    end
    for n = 1:N-1;
        if ~isequal(size(init{n}),[In(n) R(n)])
            error('Factor{%d} is the wrong size',n);
        end
    end
    if ~isequal(size(init{end}),[R(1:N-1) In(N)] )
        error('Core is the wrong size');
    end
    A = init(1:end-1);
    G = init{end};
else
    switch init(1:4)
        case 'rand'
            A = arrayfun(@rand,In(1:N-1),R(1:N-1),'uni',0);
            A{N} = eye(In(N));
            G = tensor(rand([R(1:N-1) In(N)]));
            
%             A = arrayfun(@rand,In,R,'uni',0);
%             G = tensor(rand(R));
        case {'nvec' 'eigs'}
            A = cell(N,1);
            for n = 1:N-1
                fprintf('Computing %d leading vectors for factor %d.\n',...
                    R(n),n);
                A{n} = nvecs(Y,n,R(n));
                %A{n} = max(eps,A{n});
            end
            G = ttm(Y, A,-N,'t');
            
        case {'fibe'}
            A = cell(N,1);
            for n = 1:N-1
                fprintf('Computing %d leading fibers for factor %d.\n',...
                    R(n),n);
                Yn = double(tenmat(Y,n));
                ind = maxvol2(Yn');
                A{n} = Yn(:,ind(1:R(n)));
                A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2)));
            end
            A{N} = speye(In(N));
            G = ttm(Y, A,-N,'t');
            
        otherwise
            error('Undefined initialization type');
    end
end
%% Powerful initialization
if orthoforce
    for n = 1 %1:N-1
        Atilde = ttm(Y, A, [-n -N], 't');
        A{n} = max(eps,nvecs(Atilde,n,R(n)));
    end
    A = cellfun(@(x) bsxfun(@rdivide,x,sum(x)),A,'uni',0);
    G = ttm(Y, A,-N, 't');
end
