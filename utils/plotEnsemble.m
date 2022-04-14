function plotEnsemble(samples, savedir, filename, paramNames, subsamp, burnin)
% PLOTENSEMBLE plots ensemble of MCMC chains with a red box that indicates the burnin
%   plotEnsemble(samples, savedir, filename, paramNames, false, burnin) plots with no downsampling
%
%   plotEnsemble(samples, savedir, filename, paramNames, subsamp, burnin) plots with no downsampling specified by supsamp

    if ~subsamp
        subsamp = 1;
    end

    [nsteps, nparams, nwalker] = size(samples);
    steps = 1:nsteps;
    for prm = 1:nparams
        figure
        for trc = 1:nwalker
            plot(steps(1:subsamp:end), samples(1:subsamp:end, prm, trc), 'LineWidth', 1,...
                'Color', [0.7, 0.7, 0.7]);
            hold on
        end
        ax = gca;
        plot([burnin, burnin], ax.YLim, 'r')
        plot([0, 0], ax.YLim, 'r')
        plot([0, burnin], ax.YLim, 'r')
        plot([burnin, 0], ax.YLim, 'r')
        plot([0, burnin], [ax.YLim(1) ax.YLim(1)], 'r')
        plot([0, burnin], [ax.YLim(2) ax.YLim(2)], 'r')
        
        xlabel('MCMC Steps');
        ylabel(paramNames{prm});

        fname = [savedir, filename, paramNames{prm}];
        datPath = [savedir, filename, paramNames{prm},'_data/'];
        relDatPath = [filename, paramNames{prm},'_data/'];

        saveas(gcf, fname, '.png')
        % cleanfigure; 
        % matlab2tikz([fname, '.tex'], 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
        %     'externalData', true);
    end
end