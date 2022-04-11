function plotMarginalPosterior(samples, ptrue, paramNames, paramNamesTex, savedir, fileprefix, thetaMap, thetaMean, fits, x_fit, priors, bounds)
% PLOTMARGINALPOSTEIOR plots the marginal posterior from samples
%   plotMarginalPosterior(samples, ptrue, paramNames, paramNamesTex, savedir, fileprefix, thetaMap, thetaMean, fits, x_fit, priors, bounds) standard usage

    % addpath for matlab2tkz
    myPath = pwd; paths = strsplit(myPath, 'Project-');
    addpath([paths{1}, 'matlab2tkz/'])

    numparam = numel(ptrue);
    samplePnts = makeSamplePnts(bounds, 400);

    postColor = [75 93 22] / 255; priorColor = [173 3 222] / 255; modeColor = 'b'; truthColor = 'k';
    fitColor = [0, 1, 36]/255; 
    for i = 1:numparam
        % histogram of samples
        [N, edges] = histcounts(samples(i,:),100,'Normalization','pdf');
        centers = edges(2:end) - ((edges(2) - edges(1))/2);
        % evaluate prior
        pd1 = pdf(priors{i}, samplePnts(i,:));
        % plots
        figure('Position', [0,0,300,350])
        prior = plot(samplePnts(i,:), pd1, 'Color', priorColor, 'LineWidth',2, 'DisplayName', 'Prior PDF'); hold on 
        post = histogram(samples(i,:),'Normalization','pdf', 'DisplayName', 'Posterior');
        % post = plot(centers, N, 'Color',postColor, 'LineWidth',2, 'DisplayName', 'Posterior');
        
        ksfit = plot(x_fit{i}, fits{i}, 'Color', fitColor, 'LineWidth', 2, 'DisplayName', 'KS Fit');
        ax = gca;
        lim = ax.YLim; 
        map = plot(thetaMap(i)*ones(1,2), lim, 'b', 'LineWidth',2, 'DisplayName', 'MAP');
        mean = plot(thetaMean(i)*ones(1,2), lim, 'b--', 'LineWidth',2, 'DisplayName', 'Mean');
        nominal = plot(ptrue(i)*ones(1,2), lim, 'k:', 'LineWidth',2, 'DisplayName', 'Nominal Value');

        xlabel([paramNamesTex{i}]); ylabel('PDF')
        % lim = ax.YLim; 
        legend('Location', 'northoutside')
        ylim(lim); xlim(bounds(i,:));

        % save png
        fname = [savedir, fileprefix, paramNames{i}, '.png'];
        saveas(gcf, fname)
        % write to latex file
        fname = [savedir, fileprefix, paramNames{i}, '.tex'];
        datPath = [savedir,fileprefix, paramNames{i},'data/'];
        relDatPath = [fileprefix, paramNames{i},'data'];

        cleanfigure; matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
         'externalData', true);
        close gcf
    end
end

function samplePnts = makeSamplePnts(bounds, N)
    nParam = size(bounds,1);
    samplePnts = zeros(nParam, N);
    for i = 1:nParam
        samplePnts(i,:) = linspace(bounds(i,1)*0.995, bounds(i,2)*1.005, N);
    end
end