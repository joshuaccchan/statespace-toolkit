% bvar.diag.specvar0 - long-run variance of the sample mean of each column of a
% matrix of MCMC draws, from the spectral density at frequency zero estimated with
% the Bartlett window of Newey and West (1987):
%   S = (gamma_0 + 2*sum_{l=1}^{L} (1 - l/(L+1))*gamma_l)/R,
% where gamma_l is the lag-l sample autocovariance with divisor R.
%
%   S = bvar.diag.specvar0(x, L)
%
%   x : R x k, one column per series (a vector is one series)
%   L : truncation lag, a nonnegative integer below R
%   S : 1 x k
%
% For a textbook discussion, see Chan (forthcoming), Section 6.5.
%
% See:
% Newey, W.K. and West, K.D. (1987). A Simple, Positive Semi-Definite,
% Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,
% Econometrica, 55(3): 703-708.
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 6.5.

function S = specvar0(x, L)
if isvector(x), x = x(:); end
[R, k] = size(x);
if ~(isscalar(L) && isnumeric(L) && isreal(L) && L >= 0 && L == fix(L) && L < R)
    error('bvar:diag:specvar0:badLag', ...
        'L must be a nonnegative integer below the number of draws (%d)', R);
end
S = zeros(1, k);
for j = 1:k
    xj = x(:,j) - mean(x(:,j));
    Sx = (xj'*xj)/R;
    for ell = 1:L
        w = 1 - ell/(L+1);                   % Bartlett weight
        gamma = (xj(1+ell:end)'*xj(1:end-ell))/R;
        Sx = Sx + 2*w*gamma;
    end
    S(j) = Sx/R;
end
end
