% bvar.diag.geweke - the convergence diagnostic of Geweke (1992) for each column of
% a matrix of MCMC draws: the difference between the means of an early and a late
% segment of the chain, over the standard error implied by the long-run variances
% of the two segment means.
%
%   [Z, pval, info] = bvar.diag.geweke(draws)
%   [Z, pval, info] = bvar.diag.geweke(draws, a, b, L)
%
%   draws : R x k, one column per scalar summary (a vector is one chain)
%   a, b  : the early segment is draws 1 to floor(a*R) and the late segment draws
%           floor(b*R) to R, with 0 < a < b < 1 (defaults .1 and .5)
%   L     : truncation lag of the Bartlett window in both segments; the default,
%           'auto', takes floor(4*(n/100)^(2/9)) for a segment of n draws, the rule
%           of thumb of Newey and West (1994)
%   Z     : 1 x k, approximately standard normal if the chain is stationary
%   pval  : 1 x k two-sided p-values
%   info  : the segments A and B, their lags LA and LB, and a and b
%
% The long-run variances are estimated by bvar.diag.specvar0. For a textbook
% discussion, see Chan (forthcoming), Section 6.5.1.
%
% See:
% Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the
% Calculation of Posterior Moments. In: J.M. Bernardo, J.O. Berger, A.P. Dawid and
% A.F.M. Smith (Eds), Bayesian Statistics 4, 169-193, Oxford University Press.
% Newey, W.K. and West, K.D. (1994). Automatic Lag Selection in Covariance Matrix
% Estimation, Review of Economic Studies, 61(4): 631-653.
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 6.5.1.

function [Z, pval, info] = geweke(draws, a, b, L)
if nargin < 2 || isempty(a), a = 0.10; end
if nargin < 3 || isempty(b), b = 0.50; end
if nargin < 4 || isempty(L), L = 'auto'; end
if isvector(draws), draws = draws(:); end
[R, k] = size(draws);
if ~(isscalar(a) && isscalar(b) && 0 < a && a < b && b < 1)
    error('bvar:diag:geweke:badSegments', 'a and b must satisfy 0 < a < b < 1');
end

A = 1:floor(a*R);
B = floor(b*R):R;
nA = length(A);
nB = length(B);

if ischar(L) || isstring(L)
    if ~strcmpi(L, 'auto')
        error('bvar:diag:geweke:badLag', 'L must be ''auto'' or a nonnegative integer');
    end
    LA = max(0, floor(4*(nA/100)^(2/9)));
    LB = max(0, floor(4*(nB/100)^(2/9)));
else
    LA = max(0, floor(L));
    LB = LA;
end
if nA < 2 || LA >= nA || LB >= nB
    error('bvar:diag:geweke:shortChain', ...
        'the segments have %d and %d draws, too few for lags %d and %d', nA, nB, LA, LB);
end

Z = nan(1, k);
pval = nan(1, k);
for j = 1:k
    xA = draws(A,j);
    xB = draws(B,j);
    mA = mean(xA);
    mB = mean(xB);
    SA = bvar.diag.specvar0(xA, LA);             % long-run variance of the mean of A
    SB = bvar.diag.specvar0(xB, LB);             % and of B
    Z(j) = (mA - mB)/sqrt(SA + SB);
    pval(j) = 2*(1 - normcdf(abs(Z(j))));
end

info.A = A;
info.B = B;
info.LA = LA;
info.LB = LB;
info.a = a;
info.b = b;
end
