function [tt_approx,apxrank] = tensor_denoising(X,decomposition_method,noise_level,rankX,Xinit)
% X is a tensor
%
%    min   rank(X)
%    st    |Y - X|_F^2 <  error_bound = noise_level^2 * numel(Y)
%
% where X is a tensor given in some format
%
% decomposition_method: 'tt_truncation' 'ttmps_ascu' 'ttmps_adcu'
% ttmps_atcu  'cpdepc' brtf tucker
%
% Phan Anh-Huy
% TENSORBOX

warning('off');


switch decomposition_method
    case 'tt_truncation'
        %                 C = 1.05;
        if ~isempty(noise_level)
            C = 1.01;
            accuracy = C^2*noise_level^2*numel(X);
            
            tt_1 = tt_tensor_denoise(X,accuracy);
            % err0 = norm(X(:) - full(tt_1))^2/norm(X(:))^2;
        else
            tt_1 = tt_tensor(X,1e-6,size(X),rankX);
        end
        tt_approx = tt_1;
        apxrank = rank(tt_approx)';
        
    case 'tt_ascu'
        if ~isempty(noise_level)
            
            %                 C = 1.05; % might be high if noise 0 dB
            C = 1.01; % might be high if noise 0 dB
            
            accuracy = C^2*noise_level^2*numel(X);
            
            tt_1 = tt_tensor_denoise(X,accuracy);
            % err0 = norm(X(:) - full(tt_1))^2/norm(X(:))^2;
            tt_approx = tt_1;
            
            
            % 2a ACULRO to data
            rankR = [];
            opts = tt_ascu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = tt_1;
            %opts.noise_level = sqrt(2) * noise_level^2*numel(X);% * prod(size(Y));
            opts.noise_level = accuracy;
            opts.printitn = 0;
            
            [tt_approx,out2a] = tt_ascu(X,rankR,opts);
            
        else
            % Decomposition with rank given
            opts = tt_ascu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = Xinit;
            %opts.noise_level = sqrt(2) * noise_level^2*numel(X);% * prod(size(Y));
            opts.printitn = 0;
            
            [tt_approx,out2a] = tt_ascu(X,rankX,opts);
            
        end
        
        apxrank = rank(tt_approx)';
        
    case 'ttmps_ascu'
        if ~isempty(noise_level)
            %                 C = 1.05; % might be high if noise 0 dB
            %C = 1.01; % might be high if noise 0 dB
            C = 1;
            
            accuracy = 1.5*C^2*noise_level^2*numel(X);
            
            tt_1 = tt_tensor_denoise(X,accuracy);
            % err0 = norm(X(:) - full(tt_1))^2/norm(X(:))^2;
            tt_approx = tt_1;
            
            accuracy = C^2*noise_level^2*numel(X);
            
            % 2a ACULRO to data
            rankR = [];
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = tt_1;
            %opts.noise_level = sqrt(2) * noise_level^2*numel(X);% * prod(size(Y));
            opts.noise_level = accuracy;
            opts.core_step = 2;
            opts.rankadjust = 1;
            opts.printitn = 0;
            opts.exacterrorbound = 0;
            [tt_approx,out2a] = ttmps_ascu(X,rankR,opts);
            
            apxrank = tt_approx.rank;
            
        else
            % 2a ACULRO to data
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.core_step = 2;
            opts.rankadjust = 1;
            opts.printitn = 0;
            opts.init = Xinit;
            [tt_approx,out2a] = ttmps_ascu(X,rankX,opts);
            
            apxrank = tt_approx.rank;
        end
        
    case {'ttmps_adcu' 'ttmps_adcu1' 'ttmps_adcu2'}
        
        step = str2double(decomposition_method(end));
        if isnan(step)
            step = 2;
        end
        
        
        if ~isempty(noise_level)
            
            % C = 1.05; % might be high if noise 0 dB
            C = 1.01; % might be high if noise 0 dB
            
            accuracy = C^2*noise_level^2*numel(X);
            
            tt_1 = tt_tensor_denoise(X,accuracy);
            % err0 = norm(X(:) - full(tt_1))^2/norm(X(:))^2;
            tt_approx = tt_1;
            
            
            % 2a ACULRO to data
            rankR = [];
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = tt_1;
            %opts.noise_level = sqrt(2) * noise_level^2*numel(X);% * prod(size(Y));
            opts.noise_level = accuracy;
            %opts.core_step = 2;
            opts.core_step = step;
            opts.printitn = 0;
            
            [tt_approx,out2a] = ttmps_a2cu(X,rankR,opts);
            
            apxrank = tt_approx.rank;
        else
            % 2a ACULRO to data
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = Xinit;
            %opts.core_step = 2;
            opts.core_step = step;
            opts.printitn = 0;
            
            [tt_approx,out2a] = ttmps_a2cu(X,rankX,opts);
            
            apxrank = tt_approx.rank;
        end
        
    case {'ttmps_atcu' 'ttmps_atcu1' 'ttmps_atcu2' 'ttmps_atcu3'}
        step = str2double(decomposition_method(end));
        if isnan(step)
            step = 2;
        end
        
        if ~isempty(noise_level)
            
            %                 C = 1.05; % might be high if noise 0 dB
            C = 1.01; % might be high if noise 0 dB
            
            accuracy = C^2*noise_level^2*numel(X);
            
            tt_1 = tt_tensor_denoise(X,accuracy);
            % err0 = norm(X(:) - full(tt_1))^2/norm(X(:))^2;
            tt_approx = tt_1;
            
            
            % 2a ACULRO to data
            rankR = [];
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.init = tt_1;
            %opts.noise_level = sqrt(2) * noise_level^2*numel(X);% * prod(size(Y));
            opts.noise_level = accuracy;
            opts.core_step = step;
            opts.printitn = 0;
            
            [tt_approx,out2a] = ttmps_a3cu(X,rankR,opts);
            apxrank = tt_approx.rank;
            
        else
            % 2a ACULRO to data
            opts = ttmps_a2cu;
            opts.compression = 0;
            opts.maxiters = 200;%maxiters;
            opts.tol = 1e-8;
            opts.core_step = step;
            opts.printitn = 0;
            opts.init = Xinit;
            [tt_approx,out2a] = ttmps_a3cu(X,rankX,opts);
            apxrank = tt_approx.rank;
        end
        
    case 'cpdepc'
        
        if ~isempty(noise_level)
            
            %  CPD decompition
            C = 1.01; % might be high if noise 0 dB
            accuracy = C*noise_level*sqrt(numel(X));
            
            opt_cp = cp_fLMa;
            opt_cp.init = [repmat({'rand'},1,20) 'nvec'];
            opt_cp.tol = 1e-8;
            opt_cp.maxiters = 1000;
            opt_cp.maxboost = 0;
            opt_cp.printitn = 0;
            
            apxrank = [];
            
            doing_estimation = true;
            Rcp = 8;
            best_result = [];
            
            data_ = tensor(X);
            while doing_estimation
                
                [tt_approx] = cp_nc_sqp(data_,Rcp,accuracy,opt_cp);
                tt_approx = normalize(tt_approx);
                tt_approx = arrange(tt_approx);
                
                err_cp = norm(data_ - full(tt_approx));
                norm_lda = norm(tt_approx.lambda(:));
                
                %%
                if err_cp <= accuracy + 1e-5
                    % the estimated result seems to be good
                    if isempty(best_result) || (Rcp < best_result(1))
                        best_tt_approx = tt_approx;
                        best_result = [Rcp err_cp norm_lda];
                    end
                    
                    if (Rcp > 1)   % try the estimation with a lower rank
                        Rcp_new = Rcp-1;
                    else
                        doing_estimation = false;
                    end
                else
                    Rcp_new = Rcp+1;
                end
                
                apxrank = [apxrank ; [Rcp  err_cp norm_lda]];
                if any(apxrank(:,1) == Rcp_new)
                    doing_estimation = false;
                end
                Rcp = Rcp_new;
            end
            
            tt_approx = best_tt_approx;
            
        else
            
            [tt_approx,apxrank] = tensor_denoising(X,'cpd',[],rankX);
        end
        %%
    case 'cpd'
        
        % estimated rank must be given  % Rcp = 8;
        Rcp = rankX;
        opt_cp = cp_fLMa;
        opt_cp.init = [repmat({'rand'},1,20) 'nvec'];
        
        opt_cp.tol = 1e-8;
        opt_cp.maxiters = 1000;
        opt_cp.maxboost = 0;
        opt_cp.printitn = 0;
        
        while Rcp >=1
            try
                
                tt_approx = cp_fastals(tensor(X),Rcp,opt_cp);
                
                if strcmp(decomposition_method,'cpdepc') && (Rcp>1)
                    % RUN ANC + fLM
                    %nc_alg = @cp_anc;
                    nc_alg = @cp_nc_sqp;
                    cp_alg = @cp_fLMa;
                    max_rank1norm = (10^2*Rcp);
                    epctol = 1e-7;epc_maxiters= 1000;
                    [tt_apprm ox,output_cpepc] = exec_bcp(tensor(X),tt_approx,cp_alg,nc_alg,3,max_rank1norm,epctol,epc_maxiters);
                end
                break
            catch
                Rcp = Rcp-1;
            end
        end
        apxrank = size(tt_approx.U{1},2);
        
    case 'brtf'
        
        fprintf('------Bayesian Robust Tensor Factorization---------- \n');
        % This method will not work if the data is relatively small
        % values ~~ 0.2.
        % Scaling the data up by a factor of 1000 may work for some
        % cases.
        mx = max(abs(X(:)));
        X(abs(X(:))<mx*1e-4) = 0;
        scaling = 1e8/mx;
        [model] = BayesRCP(X*scaling, 'init', 'ml', ...
            'maxRank',5*max(size(X)), 'maxiters', 50, 'initVar',scaling*noise_level^2,...
            'updateHyper', 'on','tol', 1e-5, 'dimRed', 1, 'verbose', 0);
        
        tt_approx = 1/scaling*ktensor(model.Z);
        apxrank = size(tt_approx.U{1},2);
        
    case 'tucker' %  TUCKER denoising
        
        if ~isempty(noise_level)
            
            C = 1.01;
            
            opt_td.init = 'nvecs';
            opt_td.noiselevel = C*noise_level;
            opts.printitn = 0;
            
            tt_approx = tucker_als_dn(tensor(X),size(X),opt_td);
            
            apxrank = size(tt_approx.core);
            
        else
            opt_td.init = 'nvecs';
            opts.printitn = 0;
            
            tt_approx = tucker_als(tensor(X),rankX,opt_td);
            
            apxrank = size(tt_approx.core);
        end
end

end