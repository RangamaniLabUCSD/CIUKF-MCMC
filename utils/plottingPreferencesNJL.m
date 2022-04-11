function plottingPreferencesNJL()
% sets plotting prefs
    N = 20;
    % set(0,'DefaultLineLineWidth',2)
    
    set(0,'defaultAxesFontSize',N)
    set(0, 'defaultLegendFontSize', N)
    set(0, 'defaultColorbarFontSize', N);
    
    set(0,'defaulttextinterpreter','latex')
    set(0, 'defaultAxesTickLabelInterpreter', 'latex')
    set(0, 'defaultLegendInterpreter', 'latex')
end