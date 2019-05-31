function [T,fitarr] = mtucker_als(X,R,opts,init)
% TENSOR BOX, v1. 2012
% Phan Anh Huy.
% init
if nargin < 4
    init = 'nvecs';
end
I = size(X);N = ndims(X);
if isa(X,'tensor')
    Weights = isnan(double(X));
    missingdata = any(Weights(:));
    X(Weights(:)) = 0; % set missing values to zeros, the gradients of the approx. tensors are acorrdingly changed.
    Weights = ~Weights;
elseif isa(X,'ktensor')
    try
        Weights = X.weights;
        missingdata = (~isempty(Weights)) && (~isscalar(Weights)) &&  (nnz(Weights) < prod(size(Weights)));
    catch
        missingdata = false;
    end
    if missingdata == true
        Weights = logical(double(Weights));
        X = tensor(full(X,0)); % better to convert to tensor instead of ktensor for missing data
    else
        Weights  = [];
    end
end


param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','random',@(x) (iscell(x)||ismember(x(1:4),{'rand' 'nvec'})));
param.addOptional('maxiters',200);
param.addOptional('dimorder',1:N);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.parse(opts);
param = param.Results;

U = cell(N,1);
dimorder = param.dimorder;
Nf = numel(dimorder);

% reorder dimorder so that the longest mode is first computed
%[foe,inddim] = sort(I(dimorder),'descend');
[foe,inddim] = sort(I(dimorder),'ascend'); % memory
dimorder = dimorder(inddim);
R = R(inddim);

if (exist('init','var') == 1) && ~isempty(init)
    if iscell(init)
        U(dimorder) = init(dimorder);
    elseif ((numel(init)>=4) && strcmp(init(1:4),'nvec'))
        %     for n = 1:Nf
        %         try
        %             U{dimorder(n)} = nvecs(X,dimorder(n),R(dimorder(n)));
        %         catch me
        %             Xn = double(tenmat(X,dimorder(n)));
        %             Xn = Xn*Xn';
        %             R(dimorder(n)) = min(rank(Xn),R(dimorder(n)));
        %             U{dimorder(n)} = nvecs(X,dimorder(n),R(dimorder(n)));
        %             U{dimorder(n)} = nvecs(X,dimorder(n),R(dimorder(n)));
        %         end
        %     end
        for n = 2:Nf
            try
                if n==2
                    %                     if R(n)<prod(I([1:n-1 n+1:N]))
                    U{dimorder(n)} = nvecs(X,dimorder(n),R(n));
                    %                     else % R(n)>= prod(I([1:n-1 n+1:N])) -> use QR decomposition
                    %                         [U{dimorder(n)},rr] = qr(X,dimorder(n),0);
                    %                     end
                else
                    if n==3
                        T = ttm(X,U{dimorder(n-1)},n-1,'t');
                    else
                        T = ttm(T,U{dimorder(n-1)},n-1,'t');
                    end
                    U{dimorder(n)} = nvecs(T,dimorder(n),R((n)));
                end
            catch me
                Xn = double(tenmat(X,dimorder(n)));
                Xn = Xn*Xn';
                R((n)) = min(rank(Xn),R((n)));
                U{dimorder(n)} = nvecs(X,dimorder(n),R((n)));
            end
        end
        
    elseif (numel(init)>=4) && strcmp(init(1:4),'rand')
        for n = 1:Nf
            if n==1
                U{dimorder(n)} = orth(randn(I(dimorder(n)),R((n))));
            else
                if n==2
                    T = ttm(X,U{dimorder(n-1)},n-1,'t');
                else
                    T = ttm(T,U{dimorder(n-1)},n-1,'t');
                end
                U{dimorder(n)} = nvecs(T,dimorder(n),R((n)));
            end
        end
        
    elseif (numel(init)>=5) && strcmp(init(1:5),'fiber')
        for n = 1:Nf
            Xn = double(tenmat(X,n));
            ind = maxvol2(Xn');
            U{dimorder(n)} = Xn(:,ind(1:R(n)));
            U{dimorder(n)} = bsxfun(@rdivide,U{dimorder(n)},sqrt(sum(U{dimorder(n)}.^2)));
        end
    elseif (numel(init)>=6) && strcmp(init(1:6),'maxvar')
        for n = 1:Nf
            T = double(tenmat(X,dimorder(n)));
            [~,idx] = sort(std(T),'descend');
            U{dimorder(n)} = orth(T(:,idx(1:min(numel(idx,R((n))+2)))));
        end
        %     elseif strcmp(init(1:5),'fnvec') && ismember(class(X),{'double' 'tensor'})
        %         eigsopts.disp = 0;
        %         for n = 1:Nf
        %             %         try
        %             dimarr = [1:n-1 n+1:N];
        %             T = sum(double(X),dimarr(end));
        %             for m = dimarr(end-1:-1:2)
        %                 T = sum(T,m);
        %             end
        %             [U{dimorder(n)},~] = eigs(T'*T,R(dimorder(n)),eigsopts);
        %             %         catch me
        %             %             Xn = double(tenmat(X,dimorder(n)));
        %             %             Xn = Xn*Xn';
        %             %             R(dimorder(n)) = min(rank(Xn),R(dimorder(n)));
        %             %             U{dimorder(n)} = nvecs(X,dimorder(n),R(dimorder(n)));
        %             %             U{dimorder(n)} = nvecs(X,dimorder(n),R(dimorder(n)));
        %             %         end
        %         end
    end
end

exdim = setdiff(1:N,dimorder);
for n = exdim
    U{n} = speye(I(n));
end
normX = norm(X);
fit = 0;fitarr = fit;
%% Main Loop: Iterate until convergence
for iter = 1:param.maxiters
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    for n = 1:Nf
        exset = dimorder([1:n-1 n+1:Nf]);
        if ~isempty(exset)
            Utilde = ttm(X, U(exset),exset,'t');
        else
            Utilde = X;
        end
        U{dimorder(n)} = nvecs(Utilde,dimorder(n),R(n));
    end
    
    % Assemble the current approximation
    core = ttm(Utilde, U, dimorder(n), 't');
    
    % Compute fit
    normresidual = real(sqrt(normX^2 - norm(core)^2));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);
    fitarr(end+1) = fit;
    if mod(iter,param.printitn)==0
        fprintf(' Iter %2d: fit = %e fitdelta = %7.1e\n', iter, fit, fitchange);
    end
    
    % Check for convergence
    if (iter > 1) && (fitchange < param.tol)
        break;
    end
    
end

% if ~issorted(inddim)
%     [foe,inddim2] = sort(inddim,'ascend');
%     U(dimorder) = U(dimorder);
% end
if all(R == 1)
    T = ktensor(double(core), U);
else
    %if numel(R) == 1
    %    R = R(ones(1,N));
    %end
    R = cellfun(@(x) size(x,2),U);
    T = ttensor(tensor(double(core),R(:)'), U);
end
end
