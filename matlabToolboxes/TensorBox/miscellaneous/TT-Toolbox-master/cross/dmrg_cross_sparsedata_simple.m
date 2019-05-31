function [y]=dmrg_cross(d,n,fun,eps,varargin)
% This implementation is without transmission of matrices Rmat between
% cores. Practical performance shows that this implementation yields lower
% approximation error than one with
%
%DMRG-cross method for the approximation of TT-tensors
%   [A]=DMRG_CROSS(D,N,FUN,EPS,OPTIONS) Computes the approximation of a
%   given tensor via the adaptive DMRG-cross procedure. The input is a pair
%   (D,N) which determines the size of the tensor (N can be either a
%   number, or array of mode sizes). FUN is the function to compute
%   a prescribed element of a tensor (FUN(IND)), or it can be vectorized to
%   compute series of elements of a tensor (see OPTIONS) To pass parameters
%   to FUN please use anonymous function handles. EPS is the accuracy
%   of the approximation.Options are provided in form
%   'PropertyName1',PropertyValue1,'PropertyName2',PropertyValue2 and so
%   on. The parameters are set to default (in brackets in the following)
%   The list of option names and default values are:
%       o nswp - number of DMRG sweeps [10]
%       o vec  - Fun is vectorized [ true | {false} ]
%       o verb - output debug information [ {true} | false ]
%       o y0   - initial approximation [random rank-2]
%       o radd - minimal rank change [0]
%       o rmin - minimal rank that is allows [1]
%       o kickrank - stabilization parameter [2]
%
%   Example:
%       d=10; n=2; fun = @(ind) sum(ind);
%       tt=dmrg_cross(d,n,fun,1e-7);
%
%  Vectorized version contributed by Prof. Le Song (http://www.cc.gatech.edu/~lsong/)
%
%
%
% TT-Toolbox 2.2, 2009-2013
%
%
%This is TT Toolbox, written by Ivan Oseledets et al.
%Institute of Numerical Mathematics, Moscow, Russia
%webpage: http://spring.inm.ras.ru/osel
%
%For all questions, bugs and suggestions please mail
%ivan.oseledets@gmail.com
%---------------------------
%Default parameters
rmin=1;
verb=true;
radd=0;
kickrank=2;
nswp=10;
y=[];
vectorized=false;
maxr = inf;
max_rxn = inf;
for i=1:2:length(varargin)-1
    switch lower(varargin{i})
        case 'nswp'
            nswp=varargin{i+1};
        case 'y0'
            y=varargin{i+1};
        case 'verb'
            verb=varargin{i+1};
        case 'rmin'
            rmin=varargin{i+1};
        case 'radd'
            radd=varargin{i+1};
        case 'vec'
            vectorized=varargin{i+1};
        case 'kickrank'
            kickrank=varargin{i+1};
        case 'maxr'
            maxr = varargin{i+1};
            
        case 'max_rxn' % Rank x dimension < max_rxn, i.e, rank of mode_n will not exceed max_rxn/size(n)
            max_rxn = varargin{i+1};
            
        otherwise
            error('Unrecognized option: %s\n',varargin{i});
    end
end

if ( numel(n) == 1 )
    n=n*ones(d,1);
end

max_rank = ceil(max_rxn./n);
max_rank = min(maxr,max_rank);

sz=n;
if (isempty(y) )
    y=tt_rand(sz,d,2);
end
if ( ~vectorized )
    elem=@(ind) my_vec_fun(ind,fun);
else
    elem = fun;
end
y=round(y,0); %To avoid overranks
ry=y.r;
[y,rm]=qr(y,'rl');
y=rm*y;
%Warmup procedure: orthogonalization from right to left of the initial
%approximation & computation of the index sets & computation of the
%right-to-left R matrix
swp=1;
rmat=cell(d+1,1);
rmat{d+1}=1;
rmat{1}=1; %These are R-matrices from the QR-decomposition.
index_array{d+1}=zeros(0,ry(d+1));
index_array{1}=zeros(ry(1),0);
r1=1;
for i=d:-1:2
    % Take core-th of size (r_i x n_i x r_{i+1})
    % and merge it with the right matrix Rmat of size (r_{i+1} x r_{i+1})
    % i.e.,   G_{i} x_3 Right_matrix
    % or  G_{i}_(1,2) * (Right_matrix \ox I_{n_i})
    cr=y{i}; cr=reshape(cr,[ry(i)*n(i),ry(i+1)]);
    cr = cr*r1; cr=reshape(cr,[ry(i),n(i)*ry(i+1)]); cr=cr.';
    [cr,rm]=qr(cr,0);       %  QR decomposition of  G_{i} x_3 Right_matrix
    [ind]=maxvol2(cr);
    
    % F = Gi_(1) * (Rmat_i \ox I_ni) = (Q * R)^T where  Q= cr is an orthogonal matrix
    r1=cr(ind,:); %  = (Q * inv(Q_(ind,:)) * Q_(ind,:) * R)^T
    cr=cr/r1;     %  Q * inv(Q_(ind,:))  forms the new core G_i
    r1=r1*rm;     %
    r1=r1.';      % R^T * Q_(ind,:)^T  forms the new Right matrix
    
    cr=cr.';
    y{i}=reshape(cr,[ry(i),n(i),ry(i+1)]); % the new core G_i
    
    % Compute Rmat_i from QR decomposition of (G_i x_3 Rmat_{i+1}) =
    % (Gi(1) (Rmat_{i+1} * ox I_ni)
    cr=reshape(cr,[ry(i)*n(i),ry(i+1)]);
    cr=cr*rmat{i+1}; cr=reshape(cr,[ry(i),n(i)*ry(i+1)]);
    cr=cr.';
    [~,rm]=qr(cr,0);
    rmat{i}=rm; %The R-matrix
    
    % Update indices
    ind_old=index_array{i+1};
    rnew=min(n(i)*ry(i+1),ry(i));
    ind_new=zeros(d-i+1,rnew);
    for s=1:rnew
        f_in=ind(s);
        w1=tt_ind2sub([ry(i+1),n(i)],f_in);
        rs=w1(1); js=w1(2);
        ind_new(:,s)=[js,ind_old(:,rs)'];
    end
    index_array{i}=ind_new;
end
%Forgot to put r1 onto the last core
cr=y{1}; cr=reshape(cr,[ry(1)*n(1),ry(2)]);
y{1}=reshape(cr*r1,[ry(1),n(1),ry(2)]);
not_converged = true;
dir = 1; %The direction of the sweep
i=1; %Current position
er_max=0;

while ( swp < nswp && not_converged )
    
    % A sweep through the cores
    %Compute the current index set, compute the current supercore
    %(right now without any 2D cross inside, but it is trivial to
    %implement). The supercore is (i,i+1) now.
    %Left index set is index_array{i}, right index set is index_array{i+2}
    %We will modify ry(i+1) at this step and use rmat{i} and rmat{i+2}
    %as "weighting" matrices for the low-rank approximation. The initial
    %approximation is simply rmax{i}*u{i}*u{i+1}*rmat{i+2} (hey!)
    %We also have to store the submatrix in the current factors
    %Then the algorithm would be as follows: Computex sets, compute
    %supercore. Compute rmax{i}*Phi*rmax{i+2} = U*V by SVD, then split
%     rm1=rmat{i}; rm2=rmat{i+2};
    cr1=y{i}; cr2=y{i+1};
    ind1=index_array{i};
    ind2=index_array{i+2};
     
    if vectorized

        % generate linear indices instead of sub-indices in big_index
        ix_left = 1;
        if ~isempty(ind1)
            ix_left = ind1(:,1)  +  (ind1(:,2:end)-1)*cumprod(n(1:i-2)');
        end

        ix_right = 1;
        if ~isempty(ind2)
            ix_right = ind2(1,:)'  +  (ind2(2:end,:)'-1)*cumprod(n(i+2:end-1)');
        end

        ix_ = bsxfun(@plus,ix_left,(0:n(i)*n(i+1)-1) * prod(n(1:i-1)));
        ix_ = bsxfun(@plus,ix_(:),(ix_right(:)'-1)*prod(n(1:i+1)));

        %     big_index  = ind2sub_full(n,ix_(:));

        score=elem(ix_(:));

    else

        big_index = [ ...
            ind1(kron(ones(n(i)*n(i+1)*ry(i+2),1),(1:ry(i))'),:), ...
            kron(kron(ones(n(i+1)*ry(i+2), 1),(1:n(i))'), ones(ry(i),1)), ...
            kron(kron(ones(ry(i+2), 1),(1:n(i+1))'), ones(ry(i)*n(i),1)), ...
            ind2(:, kron((1:ry(i+2))', ones(ry(i)*n(i)*n(i+1),1)))' ...
            ];

        score=elem(big_index);
    end
    
    
    %% This method need not plug in the rmax matrices
    evd_size = [ry(i)*n(i),n(i+1)*ry(i+2)];

    %Do the SVD splitting (later on we can replace it by cross for large
    %mode sizes)
    score=reshape(score,[ry(i)*n(i),n(i+1)*ry(i+2)]);
    
    
    max_r2 = min(min(evd_size),max_rank(i));
    
    if issparse(score) && (prod([ry(i)*n(i),n(i+1)*ry(i+2)])<150000  || ...
            max_r2 >= 0.6*min([ry(i)*n(i),n(i+1)*ry(i+2)]))
        score = full(score);
    end
    
    if issparse(score)
        [u,s,v]=svds(score, max_r2);
    else
        [u,s,v]=svd(score,'econ');
    end
    
    s=diag(s);
    r=my_chop2(s,norm(s)*eps/sqrt(d-1)); %Truncation
    r = min(r,max_rank(i));
    
    u=u(:,1:r); v=v(:,1:r); s=diag(s(1:r));
        
    
    %% Kick rank
    if ( dir == 1 )
        %         v=v*diag(s);
        v = v * s';
        
        ur=randn(size(u,1),kickrank);
        u=reort(u,ur);
        radd=size(u,2)-r;
        if ( radd > 0 )
            vr=zeros(size(v,1),radd);
            v=[v,vr];
        end
        r=r+radd;
    else
        %          u=u*diag(s);
        u = u * s;
        
        vr=randn(size(v,1),kickrank);
        v=reort(v,vr);
        radd=size(v,2)-r;
        if ( radd > 0 )
            ur=zeros(size(u,1),radd);
            u=[u,ur];
        end
        r=r+radd;
    end
    
    v=v';
    
    %     else
    %         [u, v] = cross2d_mat(score, eps,min([ry(i)*n(i),n(i+1)*ry(i+2) maxr]));
    %         r = size(u,2);
    %     end
    
    %     size(v)
    
    %Compute the previous approximation
    appr=reshape(cr1,[numel(cr1)/ry(i+1),ry(i+1)])*reshape(cr2,[ry(i+1),numel(cr2)/ry(i+1)]);
%     appr=reshape(appr,[ry(i),n(i)*n(i+1)*ry(i+2)]);
%     appr=rmat{i}*appr;
%     appr=reshape(appr,[ry(i)*n(i)*n(i+1),ry(i+2)]);
%     appr=appr*rmat{i+2};
    er_loc=norm(score(:)-appr(:))/norm(score(:));
    er_max=max(er_max,er_loc);
    if ( verb )
        fprintf('swp=%d block=%d new_rank=%d local_er=%3.1e\n',swp,i,r,er_loc);
    end
    ry(i+1)=r;    
    
%     u0 = u; v0 = v;    
%     u = reshape(u,[ry(i),n(i)*r]);
%     u = rmat{i}\u; %Hope it is stable blin
%     v=reshape(v,[r*n(i+1),ry(i+2)]);
%     u=reshape(u,[ry(i)*n(i),ry(i+1)]);
%     v=v/rmat{i+2}; v=reshape(v,[r,n(i+1)*ry(i+2)]);

    if ( dir == 1 )
        [u,rm]=qr(u,0);
        ind=maxvol2(u);
        r1=u(ind,:);
        u=u/r1; y{i}=reshape(u,[ry(i),n(i),ry(i+1)]);
        r1=r1*rm;
        v=r1*v; y{i+1}=reshape(v,[ry(i+1),n(i+1),ry(i+2)]);
        
%         %Recalculate rmat
%         u1=reshape(u,[ry(i),n(i)*ry(i+1)]);
%         u1=rmat{i}*u1;
%         u1=reshape(u1,[ry(i)*n(i),ry(i+1)]);
%         [~,rm]=qr(u1,0);
%         rmat{i+1}=rm;

        %Recalculate index array
        ind_old=index_array{i};
        ind_new=zeros(ry(i+1),i);
        for s=1:ry(i+1)
            f_in=ind(s);
            w1=tt_ind2sub([ry(i),n(i)],f_in);
            rs=w1(1); js=w1(2);
            ind_new(s,:)=[ind_old(rs,:),js];
        end
        index_array{i+1}=ind_new;
        if ( i == d - 1 )
            dir = -dir;
        else
            i=i+1;
        end
        
    else %Reverse direction
        v=v.'; %v is standing
        [v,rm]=qr(v,0);
        ind=maxvol2(v);
        r1=v(ind,:);
        v=v/r1; v2=reshape(v,[n(i+1),ry(i+2),ry(i+1)]); y{i+1}=permute(v2,[3,1,2]);
        r1=r1*rm; r1=r1.';
        u=u*r1; y{i}=reshape(u,[ry(i),n(i),ry(i+1)]);
        
%         %Recalculate rmat
%         v=v.';
%         v=reshape(v,[ry(i+1)*n(i+1),ry(i+2)]);
%         v=v*rmat{i+2};
%         v=reshape(v,[ry(i+1),n(i+1)*ry(i+2)]); v=v.';
%         [~,rm]=qr(v,0);
%         rmat{i+1}=rm;

        %Recalculate index array
        ind_old=index_array{i+2};
        ind_new=zeros(d-i,ry(i+1));
        for s=1:ry(i+1);
            f_in=ind(s);
            w1=tt_ind2sub([n(i+1),ry(i+2)],f_in);
            rs=w1(2); js=w1(1);
            ind_new(:,s)=[js,ind_old(:,rs)'];
        end
        index_array{i+1}=ind_new;
        if ( i == 1 )
            dir=-dir;
            swp = swp + 1;
            if ( er_max < eps )
                not_converged=false;
            else
                er_max=0;
            end
        else
            i=i-1;
        end
    end
end
return
end


%%
function val = my_vec_fun(ind, fun)
%Trivial vectorized computation of the elements of a tensor
%   [VAL]=MY_VEC_FUN(IND,FUN) Given a function handle FUN, compute all
%   elements of a tensor given in the index array IND. IND is a M x d
%   array, where M is the number of indices to be computed.
% M = size(ind, 1);
% val = zeros(1, M);
% for i = 1:M
%     ind_loc = ind(i,:); ind_loc=ind_loc(:);
%     %   ind_loc = ind(:,i);
%     val(i) = fun(ind_loc);
% end
val = fun(ind);
return
end