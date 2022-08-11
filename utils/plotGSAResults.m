function plotGSAResults(GSAResults, savedir, fileprefix, paramNames, qoiNames)
    totalInd = GSAResults.Results.Total
    firstOrderInd = GSAResults.Results.FirstOrder

    % total order
    for qoi = 1:numel(qoiNames)
        figure
        [sorted, idx] = sort(totalInd(:,qoi),1,'descend');
        bar(sorted);
        xticks(1:numel(paramNames));
        xticklabels(paramNames(idx));
        ylabel('Total Order Index');
        title(qoiNames{qoi})

        fname = [savedir, fileprefix, qoiNames{qoi},'totalOrder.tex'];
        datPath = [savedir,fileprefix, qoiNames{qoi}, 'totalOrder_data/'];
        relDatPath = [fileprefix, qoiNames{qoi}, 'totalOrder_data'];

        saveas(gcf, [savedir, fileprefix, 'totalOrder.png'])
        % cleanfigure; 
        % matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
        %     'externalData', true);

        % first order
        figure
        [sorted, idx] = sort(firstOrderInd(:,qoi),1,'descend');
        bar(sorted);
        xticks(1:numel(paramNames));
        xticklabels(paramNames(idx));
        ylabel('First Order Index');
        title(qoiNames{qoi})

        fname = [savedir, fileprefix, qoiNames{qoi},'firstOrder.tex'];
        datPath = [savedir,fileprefix, qoiNames{qoi}, 'firstOrder_data/'];
        relDatPath = [fileprefix, qoiNames{qoi}, 'firstOrder_data'];
        saveas(gcf, [savedir, fileprefix, 'firstOrder.png'])
        % cleanfigure; 
        % matlab2tikz(fname, 'standalone', true, 'dataPath', datPath, 'relativeDataPath',relDatPath,...
        %     'externalData', true);
    end
end