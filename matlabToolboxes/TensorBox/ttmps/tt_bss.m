function [A,xhat,x_tt,error] = tt_bss(y,R,sz_x,rank_x,toeplitz_order,Fs,opts)
%
% Blind source separation of sources which have TT-low rank representation
% Y: mixtures: "m" mixtures x data length
%
%   Y = A * X
%
%  R: number of sources of X, number rows of X,
%     each source should be of low-rank TT-tensor.
%
%  Y can have only one mixture.
%
%  sz_x:  size of the tensor tensorized from each source.
%
%
% PHAN ANH HUY, Mar. 2016
%
%

[m,len] = size(y);
I1 = sum(rank_x(:,2));
dx = log2((len + toeplitz_order-1)/(2*I1+toeplitz_order-2));

% Tensor expansion and tensorization
% Toeplitzation for each mixture

toepsize = [I1 ones(1,toeplitz_order-2) I1]*2^dx;
ftensor_expansion = toeplitzation(toepsize);
yx = ftensor_expansion.A(y');
yx = yx(:);

%% TT-separation
% Initialization

% Locating peaks on spectra of the observed signal
freq_x = spec_peaks(y,R,Fs);
x_tt = gen_sources(R,freq_x,Fs,len,ftensor_expansion,sz_x,rank_x);

%% Fit a TT-tensor to data y
sz = sz_x;
if m ~= 1
    sz = [sz m];
end


%% Separation of Y

opts = tt_separation;
opts.tt_fit_type = 'truncatedsvd';%'none';
opts.tol = 1e-10;


opts.maxiters = 50;
xhat_old = [];
ytt = [];
curr_err = [];
refine_counter = 0;
error  = [];

while 1
    
    opts.init = x_tt;    
    if isempty(ytt)
        [x_tt,err,Ah,ytt] = tt_separation(reshape(yx,sz),R,rank_x,opts);
    else
        [x_tt,err,Ah] = tt_separation(ytt,R,rank_x,opts);
    end    
    error = [error; err(:)];
    %% Reconstruct the estimated signals from TT-tensor and Toeplitz structure
    
    xhat1 = cell2mat(cellfun(@(x) full(x),x_tt(:)','uni',0));
    xhat = ftensor_expansion.At(xhat1);
    
    %%
    if ~isempty(xhat_old)
        [msae1_ref,msae2_ref,sae_x1_ref,sae_x2_ref] = SAE({xhat},{xhat_old});
        fprintf('MSAE of X: %s \n',sprintf('%.2f dB, ',-10*log10(sae_x1_ref)))
        
        if -10*log10(msae1_ref) > 90
            break
        end
    end
    xhat_old = xhat;
    
    new_error = err(end);
    %% refine x_tt from xhat
    refine = false;
    if refine == true
        for r = 1:R;
            xr = xhat1(:,r);
            xr = xr + norm(xr)/sqrt(numel(xr)) * .01* randn(size(xr));
            xr = tt_tensor(reshape(xr,sz_x),1e-6,sz_x);
            xr = round(xr,1e-10,rank_x(r,:));
            x_tt{r} = xr;
        end
    end
    
    %% eliminate components if there spectrums are similar
    % spectral of estimated signals
    
    if ~isempty(curr_err) && abs(curr_err - new_error)<1e-5 *curr_err
        
        if refine_counter <= 5
            
            refine_counter = refine_counter +1;
            
            for r = 1:R
                freq_x(r) = spec_peaks(xhat(:,r)',1,Fs);
            end
            [freq_x_s,ifx] = sort(freq_x);
            rep_fix = find(abs(diff(freq_x_s)) <= Fs/len);
            
            if ~isempty(rep_fix)
                eliminated_components = ifx([rep_fix rep_fix+1]);
                for k = eliminated_components
                    x_tt{k} = [];
                end
                xhat_old = [];
            end
        else
            break
        end 
    end
    
    curr_err = new_error;
end
A = Ah;    


end


function freq_x = spec_peaks(y,R,Fs)
len = size(y,2);
yf = fft(y,[],2);
frq = linspace(Fs/len,Fs/2,len/2);
powers = 20*log10(abs(yf(1:end/2)));

powers(powers<max(0,mean(powers))) = 0;
[ps,ll,w] = findpeaks(powers,'MinPeakHeight',median(powers));

[ps,ix] = sort(ps,'descend');
freq_x = frq(ll(ix(1:min(numel(ix),R))));

end


function x_tt = gen_sources(R,freq_x,Fs,len,ftensor_expan,sz_x,rank_x)
%% generate initial values based on peaks of spectral
%  R   :    number of sources
%  freq_x:  frequencies to generate the sources
%  Fs:      sampling frequency
%  len :    signal length
%  ftensor_expan:  operator to expand data, which can be toeplitzation.
%  sz_x :   size of tensor after tensorization.
%

x_tt = cell(R,1);
t = 0:len-1;
for r = 1:numel(freq_x)
    %         try
    %             xr = tt_sin_cos(log2(len),2*pi*freq_x(r)/Fs,0,0);
    %             xr = reshape(xr,szx);
    %         catch
    xr = sin(2*pi*freq_x(r)/Fs * t);
    %           end
    
    % tensor expansion if needed
    xr = ftensor_expan.A(xr(:));
    xr = tt_tensor(reshape(xr,sz_x),1e-6,sz_x);
    xr = round(xr,1e-10,rank_x(r,:));
    x_tt{r} = xr;
end
end