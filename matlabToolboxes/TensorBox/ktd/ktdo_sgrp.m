%% KTD_signle_group
function [Yh,output,Yh_p] = ktdo_sgrp(Y,Patch_Opts,Yh,options)
% Yh is initial value for the single group KTD to Y

solver = Patch_Opts.solver;

SzY = size(Y);

% Main loop to update approximation terms while fixing the other ones
% Check decomposition of approximation term
% Low-rank Matrix factorization or low-rank CPD
% Ell-1 minimization on sparse-induced transform domain
NoComps = Patch_Opts.NoComps;
patch_size = Patch_Opts.Size;
kron_unfd_ind = kron_unfoldingN(reshape(1:numel(Y),SzY),patch_size);


%% Kron-unfolding to the residue or equipvalent to the data Y because
% E = Y - sum(Y_bas) and <Y_bas,Y_est> = 0
Y_krunfold = reshape(Y(kron_unfd_ind),size(kron_unfd_ind));

% Unfolding the reference if it is given
if ~isempty(options.Yref) && any(options.Yref(:))
    options.Yref = options.Yref(kron_unfd_ind,:); % Kronecker unfolding to the orthogonal complement
else
    options.Yref = [];
end
options.tau = Patch_Opts.Regularized_par;
options.kron_unfolding_size = [];


%% Main part of the algorithm

if numel(patch_size) == 2
    % matrix approximation
    
    %% Regularized parameter for approximation term
    if ~isempty(Yh)
        options.init = reshape(Yh(kron_unfd_ind),size(kron_unfd_ind));
    else
        options.init = [];
    end
    
    
    % Patch_struct = struct('Size',[],'Transform','none','No_comps',[],'Constraints',[],'Regularized_par',[]);
    switch Patch_Opts.Constraints
        case 'lowrank'
            %options.nonnegativity = true;
            %options.A = @(x) x;
            
            %% Smooth nuclear norm if sigma noise is given
            % If epsilon is given then solving the following problem
            %   min  \|X\|_*  + mu |  X - X0 |_F^2
            %   subject to   |Y - X|_F^2 <= epsilon
            
            if ~isempty(options.epsilon) && ~isinf(options.epsilon) && ~strcmp(solver,'nc')
                
                % Scaling the data
                mx = std(Y_krunfold(:));
                Y_krunfold = Y_krunfold/mx;
                
                % Ykr(abs(Ykr(:)-mean(Ykr(:)))< std(Ykr(:))*1e-2) = 0;
                epsilon = sqrt(nnz(Y_krunfold))*options.epsilon/mx;
                
                options.maxIts = options.maxiters;  % 'continuation',false);
                
                %Omega = {size(Y_krunfold,1) size(Y_krunfold,2) (1:numel(Y_krunfold))'};
                Omega = {size(Y_krunfold,1) size(Y_krunfold,2)};
                if isempty(options.Yref)
                    [Yh, infos, opts ] = solver_sNuclearBPDN_2(Omega,Y_krunfold(:), epsilon, options.tau,Y_krunfold,[],options,NoComps);
                else
                    [Yh, infos, opts ] = solver_sNuclearBPDN_conj(Omega,Y_krunfold(:), epsilon, options.tau,Y_krunfold,[],options.Yref,options,NoComps);
                end
                objectiv = infos.f;
                Yh = Yh*mx;
                
                infos.iter = infos.niter;
                infos.f1 = 0;
                infos.f2 = norm(Y_krunfold(:) - Yh(:))^2;
                
                
            else % low rank minmization
                 
                options.maxrank =  NoComps;
                [Yh, infos,objectiv] = lowrank_approx_conj(Y_krunfold,options); % low rank for this part Y_k
            end
            
            
        case 'sparse' % sparse maximimization on the approximation term
            
            % If the noise level is given, and reference is not given,
            % OMP is used to find the sparsest solution
            % min  \| Phi(x)\|_1    subject to \| y - x \|_2 <= epsilon
            % Phi is an invirtible transformation
            
            options.A = Patch_Opts.TF;
            options.At = Patch_Opts.iTF;
            
            if ~isempty(options.epsilon) && isempty(options.Yref)
                
                switch solver
                    case 'omp'
                        
                        % adjust the noise std
                        if ischar(Patch_Opts.Transform) && ...
                                any(cell2mat(regexp({'db' 'coif'},Patch_Opts.Transform(1:2),'once')))
                            options.epsilon = options.epsilon/5;
                        end
                        
                        options.reduceDC = 1;
                        [Yh, infos,objectiv] = sparsify_orth_transform(Y_krunfold,options);
                        
                    case 'lasso'
                        % Lasso with conjugation condition  min \|y-A*x\|^2 + lambda * |x|_1
                        %options.nonnegativity = false;
                        
                        [Yh,infos,objectiv] = lasso_conj(Y_krunfold,options); % sparse constraints for this part Y_k
                        
                    case 'bpdn'
                        [Yh,infos,objectiv] = bpdn_conj(Y_krunfold,options);
                end
                
                
                
            else % use BPDN or lasso
                switch solver
                    
                    case 'lasso'
                        % Lasso with conjugation condition  min \|y-A*x\|^2 + lambda * |x|_1
                        %options.nonnegativity = false;
                        
                        [Yh,infos,objectiv] = lasso_conj(Y_krunfold,options); % sparse constraints for this part Y_k
                        
                    case 'bpdn'
                        [Yh,infos,objectiv] = bpdn_conj(Y_krunfold,options);
                        
                end
                
            end
    end
    Yh(kron_unfd_ind) = Yh(:);
    Yh = reshape(Yh,SzY);
    Yh_p = Yh;
    
    try
        Fk_val = infos.f1_val; % nuclear norm or ell-1 norm
        Res_norm = infos.f2_val;
    catch
        Fk_val = 0;
        Res_norm = norm(Y(:) - Yh(:));
    end
    
    output = struct('Res_norm',Res_norm,'Fk_val',Fk_val,'Iter',infos.iter,'Objectiv',objectiv);
    
    
else % tensor decomposition
    
    lowranktensorapproximation;
    
end


    function lowranktensorapproximation 
        % rank of approximation term
        Y_krunfold = tensor(Y_krunfold);
        
        if isempty(options.normX)
            normY = norm(Y(:));
            options.normX = normY;
        end
        
        %% Fit a single term KTD to the residue tensor E_kp
        % initialization by previous estimate
        
        % CENTRALIZE
        SzYkr = size(Y_krunfold);
        Y_krunfold = reshape(double(Y_krunfold),SzYkr(1),[]); % along the shift mode
        Ymm = mean(Y_krunfold,1);
        Y_krunfold = bsxfun(@minus,Y_krunfold,Ymm);
        Y_krunfold = tensor(reshape(Y_krunfold,SzYkr));
        
        %%
        if ~isempty(Yh)
            options.init =  Yh.U;
            options = cpcj_fastals(options);
            NoComps = size(Yh.U{1},2);

        else % initialization using CP methods
            % Initialize
            opts = cp_init;
            
            
            if all(NoComps > size(Y_krunfold))
                opts.init = {'rand' 'rand' 'rand'};
            else
                opts.init = {'nvecs' 'dtld' 'rand' 'rand' 'rand'};
            end
            
            U = cp_init(Y_krunfold,NoComps,opts);
            options.init = U;
            options = cpcj_fastals(options);
        end
        
        [Yh_p,output] = cpcj_fastals(Y_krunfold,NoComps,options);
        
        %err = (1-output_k.Fit(:,2))*normE_kp;
        %plot(err)
        %drawnow
        
        Yh = double(full_largescale(Yh_p));
        Yh = bsxfun(@plus,Yh,reshape(Ymm,[1 SzYkr(2:end)]));
        
        % Kron-folding to the estimate
        %Yh = kron_foldingN(Yh,patch_size);
        Yh(kron_unfd_ind) = Yh(:);
        Yh = reshape(Yh,SzY);
        
        % Error or cost function value
        Res_k_norm = ((1-output.Fit(end))*options.normX)^2; % |Y - Y_k|_F
        
        bias_corr = 0;
        if ~isempty(options.Yref)
            Y_mk = sum(options.Yref,2);
            bias_corr = - 2 * double(Y_krunfold(:))'*Y_mk(:) + norm(Y_mk)^2;
            Res_norm = Res_k_norm + bias_corr;
        else
            Res_norm = Res_k_norm ;
        end
        
        objectiv = ((1-output.Fit(:,2))*options.normX);
        objectiv = objectiv.^2 + bias_corr;
        
        
        output = struct('Res_norm',Res_norm,'Fk_val',0,'Iter',size(output.Fit,1),'Objectiv',objectiv);
    end
end
