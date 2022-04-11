function plotDynamicsPosterior(samples, numSamp, t, tfine, y, x0, model, savedir, filename, ptrue, thetaMap,thetaMean, hasdata, state_names, quantiles, xpost, scaling, sepStateFigs, state_colors)
% PLOTDYNAMICSPOSTEIOR plots the approximate posterior from a ensemble simulaiton
%   plotDynamicsPosterior(samples, numSamp, t, tfine, y, x0, model, savedir, filename, ptrue, thetaMap,thetaMean, hasdata, state_names, quantiles, xpost) plots individual figures for each state variable with no scaling
%
%   plotDynamicsPosterior(samples, numSamp, t, tfine, y, x0, model, savedir, filename, ptrue, thetaMap,thetaMean, hasdata, state_names, quantiles, xpost, scaling) plots individual figures for each state variable with scaling
%   plotDynamicsPosterior(samples, numSamp, t, tfine, y, x0, model, savedir, filename, ptrue, thetaMap,thetaMean, hasdata, state_names, quantiles, xpost, scaling, sepStateFigs, state_colors) plots figures as specified by sepStateFigs for each state variable with scaling and each state colored acoridng to state_colors

if nargin< 17
    scaling = ones(size(x0));
end
if nargin < 18
    sepStateFigs = true;
    state_colors = [];
end
if nargin == 18 & ~sepStateFigs
    error('Not enough input args')
end

% addpath for matlab2tkz
myPath = pwd; paths = strsplit(myPath, 'Project-');
addpath([paths{1}, 'matlab2tkz/'])

numStates = numel(x0);
green = [75 93 22] / 255; purple = [173 3 222] / 255;


quantl = quantile(xpost./scaling, quantiles(1), 3); quantu = quantile(xpost./scaling, quantiles(2), 3);
lower = [tfine'; flipud(tfine')]; upper = [quantl, fliplr(quantu)];

[~, xtrue] = ode15s(@(t,x) model(t, x, ptrue), tfine, x0);
[~, xmap] = ode15s(@(t,x) model(t, x, thetaMap), tfine, x0);
[~, xmean] = ode15s(@(t,x) model(t, x, thetaMean), tfine, x0);

xtrue = xtrue./scaling';
xmap = xmap./scaling';
xmean = xmean./scaling';

if sepStateFigs
    state_color = green; data_color = [0 0 0]; map_color = green; mean_color = green; post_color = green;
    true_color = green;
    for i = 1:numStates
        figure('Renderer', 'painters', 'Position', [0,0,800, 400])
        legEntry = ['[', num2str(quantiles(1)), ', ', num2str(quantiles(2)), '] quantile interval'];
        xposts = fill(lower, upper(i,:), post_color, 'FaceAlpha', 0.1, ...
            'DisplayName', legEntry, 'EdgeColor', post_color); hold on
        
        plot(nan, nan, 'Color', purple, 'DisplayName', 'Samples');

        plot(tfine, xtrue(:,i), 'Color', true_color, 'LineWidth', 2, ...
            'DisplayName', ['Truth']); hold on
        plot(tfine, xmap(:,i), '--', 'Color', map_color, 'LineWidth', 2, ...
            'DisplayName', ['MAP']); hold on
        plot(tfine, xmean(:,i), ':', 'Color', mean_color, 'LineWidth', 2, ...
            'DisplayName', ['MEAN']); hold on
        
        if hasdata(i)
            plot(t, [x0(i)/scaling(i) y(i,:)./scaling(i)], '.', 'Color', data_color, 'MarkerSize', 10, ...
                'DisplayName', ['noisy data']); hold on
        end

        legend('Interpreter', 'Latex', 'FontSize', 16, 'location', 'northoutside');
        xlabel('Time', 'Interpreter', 'latex'); ylabel(['$', state_names{i}, '$ Value'],'Interpreter', 'latex')
        
        % save png
        fname = [savedir, filename, '_', state_names{i}, '.png'];
        saveas(gcf, fname)
        % write to latex file
        fname = [savedir, filename, '_', state_names{i}, '.tex'];
        datPath = [savedir,filename, state_names{i},'data/'];
        relDatPath = [filename, state_names{i},'data'];
        matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
            'externalData', true);
        close gcf
    end
else 
    figure('Renderer', 'painters', 'Position', [0,0,800, 400])
    data_color = [0 0 0];
    xposts = gobjects(numStates,1); xtrues = gobjects(numStates,1); xmaps = gobjects(numStates,1); 
    xmeans = gobjects(numStates,1); data = gobjects(sum(hasdata),1) ; % empty arrays to store plot handles for legend org
    for i = 1:numStates
        state_color = state_colors(i,:);
        legEntry = ['[', num2str(quantiles(1)), ', ', num2str(quantiles(2)), '] quantile interval'];
        xposts(i) = fill(lower, upper(i,:), state_color, 'FaceAlpha', 0.1, ...
            'DisplayName', ['$', state_names{i}, '$', legEntry]); hold on
        xtrues(i) =plot(tfine, xtrue(:,i), 'Color', state_color, 'LineWidth', 2, ...
            'DisplayName', ['$', state_names{i},'$  true']); hold on
        xmaps(i) = plot(tfine, xmap(:,i), '--', 'Color', state_color, 'LineWidth', 2, ...
            'DisplayName', ['$', state_names{i},'$  MAP']); hold on
        xmeans(i) = plot(tfine, xmean(:,i), ':', 'Color', state_color, 'LineWidth', 2, ...
            'DisplayName', ['$', state_names{i},'$  MEAN']); hold on
        
        if hasdata(i)
            data(i) = plot(t, [x0(i) y(i,:)], '.', 'Color', state_color, 'MarkerSize', 10, ...
                'DisplayName', ['$', state_names{i},'$  data']); hold on
        end
    end
    a = vertcat(xposts, xtrues, xmaps, xmeans, data)
    legend(a, 'Interpreter', 'Latex', ...
            'FontSize', 16, 'location', 'westoutside');
    xlabel('Time', 'Interpreter', 'latex'); ylabel('State Vairable Value','Interpreter', 'latex')

    % save png
    fname = [savedir, filename, '.png'];
    saveas(gcf, fname)
    % write to latex file
    fname = [savedir, filename, '.tex'];
    datPath = [savedir,filename, '_data/'];
    relDatPath = [filename, state_names{i},'data'];
    matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
        'externalData', true);
    close gcf
end
end
