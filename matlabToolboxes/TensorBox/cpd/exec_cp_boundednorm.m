function [Pb,output] = exec_cp_boundednorm(Y,Pb,cp_alg,maxiters,lambda_normbound,nc_alg)
% This file executes the bounded CP decomposition of the tensor T with
% initial Pb or the current point approximated of T
%
%    min    \|Y - P\|_F^2 
%  subject to   sum_n  \lambda_n^2  <= lambda_normbound^2
%
%  P is a ktensor of U1, ..., UN
%  
if nargin <3
    cp_alg = @cp_boundlda_sqp;  %@cp_boundlda_ipm @cp_boundlda_als
end
if nargin < 4
    maxiters = 1000;
end
if nargin < 5
    lambda_normbound = 1e4;
end

if nargin < 6
    nc_alg = @cp_nc_sqp;  %Norm reservation correction method s
end

Y = tensor(Y);
normY = norm(Y);
N = ndims(Y);


%% STAGE 1: initialize paramters and run ALS in a few iterations (10, 50  iterations)
out_s1 = [];
if isa(Pb,'ktensor')
    % an initial tensor is given 
    P_s1 = Pb;clear Pb;
    % Pb is given as an intial point 
    Rx = size(P_s1.u{1},2);
    
else
    % Generate an initial tensor 
    % Pb is the rank Rx
    Rx = Pb;
    opts = cp_fLMa;
    opts.init = [ 'dtld' 'nvec'  'nvec' repmat({'rand'},1,100)];
    % opts.init = repmat({'rand'},1,10);
    opts.printitn = 1;
    opts.maxiters = 10;
    %opts.linesearch = 0;
    % opts.maxiters = 50;
    opts.TraceRank1Norm = true;
    opts.tol = 1e-9;
 
    if Rx <= 40
        [P_s1,out_s1] = cp_fLMa((Y),Rx,opts);
    else
        [P_s1,out_s1] = cp_fastals((Y),Rx,opts);
    end
    
    % result in the first stage
    U = cellfun(@(x) bsxfun(@times,x,P_s1.lambda'.^(1/N)),P_s1.u,'uni',0);
    P_s1 = ktensor(U);     
end

%% STAGE 2: Correct the initial tensor with the norm correction of rank-1 tensors
%  i.e., find an equipvalent tensor with a smaller lambda
%       min          |Lambda|^2
%
%     subject to    |Y - P|_F < delta
%  P is a ktensor of U1, ..., UN

Pb = P_s1;
% delta = |Y-Yx|_F
% 
delta = max(0,normY^2 + norm(Pb)^2 - 2*innerprod(Y,Pb));
delta = sqrt(delta);
% set the error bound to a bit higher than the approximation error
delta = 1.1*delta;

opts = cp_anc;
opts.maxiters = 10;
opts.printitn = 1;
opts.linesearch = 0;
opts.tol = 1e-8;

opts.init = P_s1;
[Pb,outputb] = cp_anc(Y,Rx,delta,opts);

% run the norm correction again 
opts = nc_alg();
opts.init = Pb;
opts.tol = 1e-8;
opts.maxiters = 1000;
opts.ineqnonlin = false;
% opts.ineqnonlin = true;

if Rx > 100
    [P_s2,output_s2] = cp_anc(Y,Rx,delta,opts);
else
    [P_s2,output_s2] = nc_alg(Y,Rx,delta,opts);
end

% output_s2.fval = [outputb.fval; output_s2.fval];

%% Step 3: Execute the BSQP algorithm for bounded CDP

Pp = normalize(P_s2);
delta = norm(Pp.lambda)^2;
deltax = delta*4; % set the bound to a higher value 

P_s3 = P_s2;

opts = cp_alg();

opts.alsinit = 0;
opts.tol = 1e-10;
opts.maxiters = 300;
opts.linesearch = 0;
opts.ineqnonlin = true;
% opts.Algorithm = 'sqp'
opts.Correct_Hessian = true; % good with the equality constraint
% opts.bound_ell2_constraint = true;  \|\theta\|_F^2 < xi
opts.bound_ell2_constraint = false;  % for the constraint \|\theta\|_F^2< xi
opts.printitn = 1;
opts.TraceRank1Norm = true;

Output_normcorrection3 = [];
current_error = normY^2 + norm(P_s3)^2 - 2*innerprod(Y,P_s3);
previous_error = current_error;
% current_error = sqrt(max(0,current_error))/normY;
P_s3_good = P_s3;

%%
for kiter = 1: maxiters
    
    
    %% Run the norm correction method if ell-2 norm of the weights of 
    %  rank-1 tensors exceed a bound 
    %
    %     min        sum_n lambda_n^2 = |\blambda|^2
    %
    %   subject to   |X - P|_F <= approximation_error
    %
    %
    if deltax >  lambda_normbound
        nc_opts = cp_anc;
        nc_opts.maxiters = 1000;
        nc_opts.printitn = 1;
        nc_opts.linesearch = 0;
        nc_opts.tol = 1e-8;
        
        nc_opts.init = P_s3;
        [Pb,outputb] = cp_anc(Y,Rx,sqrt(current_error),nc_opts);
        
        
        % Run again the Norm correction method
%         nc_alg = @cp_nc_itp;
        nc_opts = nc_alg();
        nc_opts.init = Pb;
        nc_opts.tol = 1e-10;
        nc_opts.maxiters = 1000;
        nc_opts.ineqnonlin = false; %  |X - P|_F = approximation_error
%         nc_opts.ineqnonlin = true; %  |X - P|_F < approximation_error
        
        if Rx > 100
            [P_s3,output_s2] = cp_anc(Y,Rx,sqrt(current_error),nc_opts);
        else
            [P_s3,output_s2] = nc_alg(Y,Rx,sqrt(current_error),nc_opts);
            %opts.Hessian = 'approximation';
        end
        %Pp = normalize(P_s3); deltax = 2*norm(Pp.lambda);
    end


    %% RUN CP or CPD with a bounded rank-1 norm
    
    %       min     \|X - P\|_F^2
    %   subjec to   sum_n  \lambda_n^2  < delta^2

    opts.init = P_s3;
    
%     try
        [Pbx,outputx] = cp_alg(Y,Rx,deltax,opts);
        
        if isfield(outputx,'Fit')
            Output_normcorrection  = ((1-real(outputx.Fit(:,2)))*normY).^2;
        else
            Output_normcorrection = 2*outputx.fval;% |y - yx|_F^2
        end
%     catch
%         %% fLM
%         [Pbx,outputx] = cp_alg(tensor(Y),Rx,opts);
%         Output_normcorrection = (1-real(outputx.Fit(:,2)));
%     end
%     if isfield(outputx,'Rank1Norm')
%         Rank1Norm_normcorrection = (sum(outputx.Rank1Norm.^2,2));
%     end
    
    P_s3 = Pbx;
    
    
    %%
    %     if abs(Output_normcorrection(end) - Output_normcorrection(end-1)) < opts.tol
    % increase delta
    %        if abs(current_error - Output_normcorrection(end))<opts.tol
    %            break
    %        else
    %            deltax = deltax*2;
    %        end
    % reduce delta every 100 iterations if the relative error decreases
    
    % if difference between two consecutive batches is smaller than tol
    % the algorithm seems to getting suck in local minima
    % increasing delta will make the algorithm to factorize the data
    % easier
    %
    if abs(current_error - Output_normcorrection(end))<=opts.tol
        deltax = deltax*1.1;
        previous_error = current_error;
        current_error = Output_normcorrection(end);
        P_s3_good = P_s3;
        Output_normcorrection3 = [Output_normcorrection3 ; Output_normcorrection];

    elseif (current_error > Output_normcorrection(end))
        % If the error decreases, seems to higher than the optimal lambda
        % if the bound delta is too high, the convergence will be slow
        % Try to reduce delta
%         if abs(current_error - Output_normcorrection(end)) <= 1e-6
%             deltax = deltax/1.01;
%         end
        previous_error = current_error;
        current_error = Output_normcorrection(end);
        P_s3_good = P_s3;
        Output_normcorrection3 = [Output_normcorrection3 ; Output_normcorrection];
        
    elseif current_error < Output_normcorrection(end)
        % if the error increases, the bound is too small.
        % a bad delta
        deltax = deltax*1.5;
        P_s3 = P_s3_good;
    end
    
    %     end
    if numel(Output_normcorrection)>5 && (abs(previous_error - current_error)/normY^2<=opts.tol)
        break
    end
end

%%
if ~isempty(out_s1)
    Output_normcorrection3 = [1-out_s1.Fit(:,2) ; sqrt(Output_normcorrection3)/normY];
end
output.fval = Output_normcorrection3; % | Y - Yx|_F^2
Pb = P_s3;

% %%
% 
% fig = figure(1);
% clf
% 
% % h(1) = loglog();
% % hold on
% h(1) = loglog(Output_normcorrection_cmb);
