function Ax = exact_seq_tt_to_cp(Y,R,SzY,method)
% Sequential conversion from a TT-tensor Xtt of rank-R to a rank-R tensor.
%   Xtt is often obtained by fit a TT-tensor to a noiseless (or nearly
%          noisefree) rank-R tensor
%   Ah : cell arrray of factor matrices of the rank-R tensor.
%
% Phan Anh Huy, 2016

if (nargin < 3) || isempty(SzY)
    SzY = size(Y);
end
if nargin < 4
    method = 'bestrank1'; % 'cpd'
end
    

if ~isa(Y,'tt_tensor')
    N = ndims(Y);

    Y = reshape(double(Y),[SzY(1)*SzY(2) SzY(3:end-2) SzY(end-1)*SzY(end)]);
    
    Xtt1 = tt_tensor(Y,1e-6,[SzY(1)*SzY(2) SzY(3:end-2) SzY(end-1)*SzY(end)],[1 R*ones(1,N-2) 1]');
    
    Xtt = round(Xtt1,1e-6,[1 R*ones(1,N-2) 1]');
    tt_opts = tt_a2cu;
    tt_opts.init = Xtt;
    [Xtt,output] = tt_a2cu(Xtt1,[1 R*ones(1,N-2) 1]',tt_opts);
    % SzY = size(Y);
else
    Xtt = Y;
    N = numel(SzY);
end


%% CPD option
%cp_func = @cp_fastals;
if R <= 20
    if isreal(Xtt{1})
        cp_func = @cp_fLMa;
    else
        cp_func = @cpx_fLMa;
    end
else
    cp_func = @cp_fastals;
end

cp_opts = cp_func();
cp_opts.maxiters = 2000;
cp_opts.init = {'nvec' 'dtld'};
cp_opts.printitn = 0;
cp_opts.maxboost = 0;


G = core2cell(Xtt);
G{1} = reshape(G{1},SzY(1),SzY(2),[]);
G{end} = reshape(G{end},[],SzY(end-1),SzY(end));

Pn_ = cell(N,1);
Ax = cell(N,1);
for n = 1:N-2
    Gnx = G{n};
    if n > 1
        %G{n} = double(ttm(tensor(G{n}),Pn_{n-1}.u{3}*diag(Pn_{n-1}.lambda),1,'t'));
        %S = Pn_{n-1}.u{3}*diag(Pn_{n-1}.lambda);
        Gnx = S.' * reshape(G{n},size(G{n},1),[]);
        Gnx = reshape(Gnx,[],size(G{n},2),size(G{n},3));
    end
    
    % CPD of the first core tensor
    if n == 1
        Pn = cp_func(tensor(Gnx),R,cp_opts);
        Ax(1:2) = Pn.u(1:2);
        S = Pn.u{3}*diag(Pn.lambda);
        
    else % best rank-1 of horitonal slices of the 2nd,..., N-2 th core tensors
        
        switch method 
            case 'bestrank1'
                Axn = zeros(SzY(n+1),R);
                if n < N-2
                    S = zeros(R,R);
                else
                    S = zeros(SzY(N),R);
                end
                for r = 1:R
                    [ar,sr,ur] = svds(squeeze(Gnx(r,:,:)),1);
                    Axn(:,r) = ar;
                    S(:,r) = conj(ur)* sr;
                end
                Ax{n+1} = Axn;
                
            case 'cpd'
                Pn = cp_func(tensor(Gnx),R,cp_opts);
                
                % permute the first factor matrix to be diagonal matrix
                [foe,ix] = max(abs(Pn.u{1}),[],2);
                
                Pn.u = cellfun(@(x) x(:,ix),Pn.u,'uni',0);
                Pn.lambda = Pn.lambda(ix);
                
                [foe,ix] = max(abs(Pn.u{1}),[],1);
                ix2 = sub2ind(size(Pn.u{1}),ix(:),(1:size(Pn.u{1},1))');
                al = Pn.u{1}(ix2);
                
                Pn.u{1} = Pn.u{1}*diag(1./al);
                Pn.lambda = bsxfun(@times,Pn.lambda,al);
                
                S = Pn.u{3}*diag(Pn.lambda);
        end
    end 
end
Ax{N} = S;

% %%
% Pn_ = cell(1,N);
% % cp_func = @cp_fastals;
% % cp_func = @cpx_fLMa;
% for n = 1:N-2
%     Gnx = G{n};
%     if n > 1
%         %G{n} = double(ttm(tensor(G{n}),Pn_{n-1}.u{3}*diag(Pn_{n-1}.lambda),1,'t'));
%         S = Pn_{n-1}.u{3}*diag(Pn_{n-1}.lambda);
%         Gnx = S.' * reshape(G{n},size(G{n},1),[]);
%         Gnx = reshape(Gnx,[],size(G{n},2),size(G{n},3));
%     end
%     
%     %
%     if n == 1
%          Pn = cp_func(tensor(Gnx),R,cp_opts);
%     else
%         Ax = zeros(SzY(n+1),R);
%         if n < N-2
%             Bx = zeros(R,R);
%         else
%             Bx = zeros(SzY(N),R);
%         end
%         for r = 1:R
%             [ar,sr,ur] = svds(squeeze(Gnx(r,:,:)),1);
%             Ax(:,r) = ar;
%             Bx(:,r) = conj(ur)* sr;
%         end
%         Pn = ktensor({eye(R), Ax, Bx});
%     end
%      
%     Pn_{n} = Pn;
% end
% 
% % extract A
% Ax = [Pn_{1}.u{1} cellfun(@(x) x.u{2},Pn_(1:N-2),'uni',0) Pn_{N-2}.u{3}*diag(Pn_{N-2}.lambda)];
% 
% %%