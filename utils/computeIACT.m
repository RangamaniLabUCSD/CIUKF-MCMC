function IACT = computeIACT(samples)
% COMPUTEIACT Integrated Autocorrelation Time (IACT) Analysis
%   IACT = computeIACT(samples) perform IACT analysis for the ensemble in samples (nParam, Nsteps, nWalkers)
%   Uses the UWerr_FFT() function from <a href="matlab:web('https://www.physik.hu-berlin.de/de/com/UWerr_fft.m')">UWerr_fft.m </a>
%

% Here we want to comput the IACT for the ensemble
    % We take the following approach
    % 1) Compute IACT for each chain for every parmeter using the UWerr_fft() function
    % - This code is authored by Ulli Wolff
    % - See paper: "MC errors with less error" for details
    % 2) Take the average over all walkers in the ensemble
    % 3) Take the maximum over the set of parameters

    % 1) Compute all IACTs:
    nParam = size(samples,2); % number of params
    Nsteps = size(samples,1); % number of MCMC steps
    nWalkers = size(samples,3); % number of walkers in the ensemble

    IACTs = zeros(nParam,nWalkers); % matrix of IACTs (nParam x nWalkers)
    for kk=1:nWalkers
        for jj=1:nParam
            % parameters here are essentially all default!
            [~,~,~,tauinttmp,~,~] = UWerr_fft(squeeze(samples(:,jj,kk)),1.5,Nsteps,1,1,1);
            IACTs(jj,kk) = tauinttmp;
        end
    end
    
    % 2) Take the average over each walker:
    meanIACTs = mean(IACTs,2);

    % 3) Take the max over all set of params
    IACT = max(meanIACTs);
    % disp(['The approx IACT is: ', num2str(IACT), '!']);
end