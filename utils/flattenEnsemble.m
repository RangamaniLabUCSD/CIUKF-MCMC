function samples2d = flattenEnsemble(samples3d)
% FLATTENENSEMBLE 
%   samples2d = flattenEnsemble(samples3d) 3d set of samples (nSamples x nParam x nChains) into a 2d matrix of samples (nParam x nSamples*nChains)
%

    dims = size(samples3d);
    samples2d = zeros(dims(2), dims(1)*dims(3)); % preallocate matrix

    for idx = 1:dims(2)
        temp = samples3d(:,idx,:);
        samples2d(idx,:) = temp(:);
    end
end