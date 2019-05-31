function [U1,U2,U3] = fast_tucker2_denoising(X,init,maxiters,tol,noiselevel,exacterrorbound)

% Simple fast Tucker-2 decomposition for denoising problem
%    min  |X - [[G;U1 U3]] |_F^2 <= noiselevel^2 * numel(X)
%
SzX = size(X);
if nargin<6
    exacterrorbound = false;
end
 
%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= 2
        error('OPTS.init does not have %d cells',2);
    end
    if ~isequal(size(Uinit{1},1),SzX(1))
        error('OPTS.init{%d} is the wrong size',1);
    end
    if ~isequal(size(Uinit{2},1),SzX(3))
        error('OPTS.init{%d} is the wrong size',2);
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(2,1);
        Uinit{2} = rand(SzX(3));
         
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        % Compute an orthonormal basis for the dominant
        % Rn-dimensional left singular subspace of
        % X_(n) (1 <= n <= N).
        Uinit = cell(2,1);
         
        fprintf('  Computing leading e-vectors for factor %d.\n',2);
        Uinit{2} = nvecs(tensor(X),3,SzX(3));
        
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U1 = Uinit{1};
U3 = Uinit{2};
printitn  = 0;
if printitn > 0
    fprintf('\nTucker-2 for denoising:\n');
end

if noiselevel>0
    accuracy = norm(X(:))^2 - noiselevel^2 * prod(SzX);
end

X1 = reshape(X,size(X,1),[]);
X3 = reshape(X,[],size(X,3));
normU2 = zeros(maxiters,1);

% The loop updates U1 and U3, while U2 is computed after the iteration
for kiter = 1:maxiters
      
    % Update U1
    T = X3*U3;
    T = reshape(T,SzX(1),[]);
    Q = T*T';
 
    U1 = minrank_eqbound(Q,accuracy);
    R(1) = size(U1,2);
    
    % Update U3
    T = U1'*X1;
    T = reshape(T,[],SzX(3));
    Q = T'*T; 
    
    U3 = minrank_eqbound(Q,accuracy);
    R(2) = size(U3,2);
     
    % Stop if converged
    normU2(kiter) = sum(s(1:r3));
    % err(kiter) = norm(X - U1 x U2 x U3)^2 
    %            = norm(X)^2 - norm(U2)^2
    if (kiter>1) && (abs(normU2(kiter) - normU2(kiter-1))<= tol)
        break
    end
end
U2 = reshape(T*U3,r1,[],r3);
end