% bvar.diag.mcse - Monte Carlo standard error of the posterior mean estimated from
% each column of a matrix of MCMC draws, the numerical standard error of Geweke
% (1992): the square root of the long-run variance of the sample mean.
%
%   MCSE = bvar.diag.mcse(draws, L)
%
%   draws : R x k, one column per scalar summary (a vector is one chain)
%   L     : truncation lag of the Bartlett window, a nonnegative integer below R;
%           take it beyond the lag at which the autocorrelations die out
%   MCSE  : 1 x k
%
% The long-run variance is estimated by bvar.diag.specvar0. For a textbook
% discussion, see Chan (forthcoming), Section 6.5.2.
%
% See:
% Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the
% Calculation of Posterior Moments. In: J.M. Bernardo, J.O. Berger, A.P. Dawid and
% A.F.M. Smith (Eds), Bayesian Statistics 4, 169-193, Oxford University Press.
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 6.5.2.

function MCSE = mcse(draws, L)
if isvector(draws), draws = draws(:); end
[R, k] = size(draws);
MCSE = zeros(1, k);
for j = 1:k
    x = draws(:,j);
    Omega = bvar.diag.specvar0(x, L)*R;           % long-run variance of the draws
    MCSE(j) = sqrt(Omega/R);
end
end
