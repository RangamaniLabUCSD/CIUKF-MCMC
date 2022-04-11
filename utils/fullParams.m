function thetaFull = fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull)
% FULLPARAMS makes a full parameter vector from a set of free parameters and the full set of nominal values
%   thetaFull = fullParams(theta, freeParamIndex, fixedParamIndex, ptrueFull)
%

    thetaFull = zeros(size(ptrueFull));
    thetaFull(fixedParamIndex) = ptrueFull(fixedParamIndex);
    thetaFull(freeParamIndex) = theta(1:numel(freeParamIndex));
end