function [xpost, index] = runEnsembleSimulation(model, jacobian, samples, t, x0, numSamp)
% RUNENSEMBLESIMULATION runs an ensemble simulation
%   [xpost, index] = runEnsembleSimulation(model, jacobian, samples, t, x0, numSamp) runs simulation with numSamp random samples form samples

    index = randsample(1:size(samples, 2), numSamp);
    xpost = zeros(numel(x0), numel(t), numSamp);

    parfor i = 1:numSamp
        if ~isempty(jacobian)
            odeopts = odeset('Jacobian', @(t,x)jacobian(t,x,samples(:, index(i))));
        else
            odeopts = odeset();
        end

        [~, xhat] = ode15s(@(t, y) model(t, y, samples(:, index(i))), t, x0, odeopts);

        if isreal(xhat)
            xpost(:, :, i) = xhat';
        else
            xpost(:, :, i) = nan;
        end
    end

end
