% bvar.diag.inefficiency_factor - inefficiency factor of each column of a matrix of
% MCMC draws: the long-run variance of the draws over their variance, which is the
% reciprocal of the relative numerical efficiency of Geweke (1992). It is the number
% of draws that carry the information of one independent draw.
%
%   IF = bvar.diag.inefficiency_factor(draws, L)
%
%   draws : R x k, one column per scalar summary (a vector is one chain)
%   L     : truncation lag of the Bartlett window, a nonnegative integer below R;
%           take it beyond the lag at which the autocorrelations die out
%   IF    : 1 x k; the effective sample sizes are R./IF
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

function IF = inefficiency_factor(draws, L)
if isvector(draws), draws = draws(:); end
[R, k] = size(draws);
IF = zeros(1, k);
for j = 1:k
    x = draws(:,j);
    sigma2 = var(x, 1);                           % variance, divisor R
    Omega = bvar.diag.specvar0(x, L)*R;           % long-run variance of the draws
    IF(j) = Omega/sigma2;
end
end
