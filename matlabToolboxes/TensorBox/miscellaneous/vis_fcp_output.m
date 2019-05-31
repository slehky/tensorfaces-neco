function vis_fcp_output(BagofOut)
% Plot approximation errors in BagofOut as function of execution times
%
% This function is a part of the TENSORBOX, 2012.
% Copyright Phan Anh Huy, 04/2012
%

fig = figure(1); clf; set(gca,'fontsize',16); hold on
clear h hrd(k)
hold on;

stylecode = {'m' 'r' 'k'};

tottime = eps;
%
try
    if strcmp(BagofOut{1}.Name,'Compress')
        finalerr = 1- BagofOut{1}.Fit(2);
    else
        finalerr = 1- outputd.Fit(1);
    end
catch
    finalerr = inf;
end
%%
for k = 1:numel(BagofOut)
    style = stylecode{1};
    switch BagofOut{k}.Name
        case 'Compress'
            style = [style ':'];
        case 'CP'
            style = [style '--'];
        case {'Rnk1' 'lowrank'}
            style = [style '-.'];
        case 'CPRef'
            style = [style '-'];
    end
    if isfield(BagofOut{k},'Fit') && ~isempty(BagofOut{k}.Fit)
        if min(size(BagofOut{k}.Fit)) > 1
            it1 = tottime + linspace(0,BagofOut{k}.Time,size(BagofOut{k}.Fit,1)+1);
            hrd(k) = plot(it1,[finalerr ; 1-real(BagofOut{k}.Fit(:,2))],style,...
                'linewidth',2);
        else
            it1 = tottime + linspace(0,BagofOut{k}.Time,numel(BagofOut{k}.Fit)+1);
            hrd(k) = plot(it1,[finalerr ; 1-real(BagofOut{k}.Fit(:))],style,...
                'linewidth',2);
        end
        finalerr = 1-real(BagofOut{k}.Fit(end)) +eps;
    else
        hrd(k) = plot([tottime tottime+BagofOut{k}.Time],[finalerr ;finalerr],style,...
            'linewidth',2);
    end
    %         if rule == 3
    if k < numel(BagofOut)
        str2 = BagofOut{k}.Name;
    else
        str2 = sprintf('%s %.2f',BagofOut{k}.Name,finalerr);
    end
    text(tottime+BagofOut{k}.Time/2,double(finalerr),str2,'fontsize',16,...
        'FontWeight','bold','verticalalignment','bottom')
    %         end
    %
    tottime = tottime + BagofOut{k}.Time;
    
    hm = plot(tottime,finalerr,'ro','markersize',14,'linewidth',2);
end



xlabel('Time (seconds)')
grid on


set(gca,'yscale','log')
axis tight
% yl = ylim;
% ylim([max(-7,yl(1)) yl(2)])

set(gca,'xMinorGrid','off','yMinorGrid','off')
ylabel('Approximation Error')

legend(hrd(end),'FCP','location','best')

% axis auto
%%
% fn = sprintf('fig_3W_errvstime_N%dI%dR%d_%ddB',R,I(1),N,SNR);
% set(fig,'PaperPositionMode','auto')
% saveas(fig,fn,'fig')
% fn = [fn '.eps'];
% saveas(fig,fn,'epsc')
% %print(gcf,fn,'-depsc','-opengl');
% fixPSlinestyle(fn,fn)
