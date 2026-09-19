%% GOLDEN-RUN PATCH (2026-09-19): a file added to the build copy, not a change to any
%% legacy file. The package's scripts call autocorr(u, nlag), the positional form of the
%% Econometrics Toolbox function they were written for, which R2025b's autocorr rejects
%% (it takes name-value arguments). This function, written for the golden runs, returns
%% what that call returned: the sample autocorrelations of u at lags 0 to nlag, from the
%% FFT of the demeaned series padded to 2^(nextpow2(n)+1) points, normalized by lag 0.
%% It sits in the package folder, so it shadows the toolbox function there only.

function acf = autocorr(u, nlag)
isrow_u = isrow(u);
u = u(:) - mean(u);
F = fft(u, 2^(nextpow2(length(u)) + 1));
r = ifft(F .* conj(F));
r = r(1:nlag+1);
acf = real(r ./ r(1));
if isrow_u
    acf = acf.';
end
end
