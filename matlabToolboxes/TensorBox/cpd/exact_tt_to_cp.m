%% 
function Ah = exact_tt_to_cp(Xtt,R)
% Exact conversion from a TT-tensor Xtt of rank-R to a rank-R tensor.
%   Xtt is often obtained by fit a TT-tensor to a noiseless (nor nearly
%          noisefree) rank-R tensor 
%   Ah : cell arrray of factor matrices of the rank-R tensor.
%
% Phan Anh Huy, 2016

G = core2cell(Xtt);
N = ndims(Xtt);
Pn_ = cell(N,1);

% Decomposition of each core G{n} by a rank-R tensor.
% The second factor matrices are factor matrices of the rank-R tensor.

if R <= 20
    cp_func = @cp_fLMa;
else
    cp_func = @cp_fastals;
end
    
cp_opts = cp_func();
cp_opts.maxiters = 2000;
cp_opts.init = {'nvec' 'dtld'};
cp_opts.printitn = 0;
cp_opts.maxboost = 0;

for n = 2: N-1
    Pn = cp_func(tensor(G{n}),R,cp_opts);
    Pn_{n} = Pn;
end

% matching order and scaling between
% Pn_{n}.u{3} and Pn_{n+1}.u{1}
% For the exact case, the following relation holds
%   Pn_{n}.u{3}' * inv(Pn_{n+1}.u{1}) = diag(dn)
%
% The following normalization normalizes Pn_{n}.u{3} *and Pn_{n+1}.u{1}
% so that
%   Pn_{n}.u{3}' * inv(Pn_{n+1}.u{1}) = I_R

for n = 2:N-2
    if numel(Pn_{n}.u)>2
        C = (Pn_{n}.u{3}'*Pn_{n+1}.u{1});
    else 
        C = Pn_{n+1}.u{1};
    end
    [foe,ix] = max(abs(C),[],2);
    
    Pn_{n+1}.u = cellfun(@(x) x(:,ix),Pn_{n+1}.u,'uni',0);
    Pn_{n+1}.lambda = Pn_{n+1}.lambda(ix);
    
    if numel(Pn_{n}.u)>2
        C = (Pn_{n}.u{3}'*Pn_{n+1}.u{1});
        al = diag(C);
        
        Pn_{n}.u{3} = Pn_{n}.u{3}*diag(1./al);
        Pn_{n}.lambda = bsxfun(@times,Pn_{n}.lambda,al);
    end    
end

% Approximate to the factor matrices of X
Ah = cell(N,1);
Ah{1} = reshape(G{1},size(G{1},2),[])*Pn_{2}.u{1} ;
if numel(Pn_{N-1}.u)>2
    Ah{N} = (squeeze(G{N})')*(Pn_{N-1}.u{3});
else
    Ah{N} = (squeeze(G{N})');
end
Ah(2:N-1) = cellfun(@(x) x.u{2} * diag(x.lambda),Pn_(2:N-1),'uni',0);

end