% ssm.intlike - observed-data (integrated) log likelihood of a linear Gaussian
% state space model, with the states integrated out, from banded precision
% matrices: the method of Chan and Jeliazkov (2009).
%
%   ll = ssm.intlike(y, Z, iR, P, b)
%   [ll, alphahat, K] = ssm.intlike(y, Z, iR, P, b)
%
%   y        : Tn x 1 stacked observations, net of any part not involving the
%              states, with y = Z*alpha + e and e ~ N(0, R)
%   Z        : Tn x Tm matrix loading the stacked states alpha
%   iR       : Tn x Tn precision inv(R), sparse (diagonal or banded)
%   P, b     : Tm x Tm prior precision and Tm x 1 prior mean of alpha implied by
%              the transition equation G*alpha = btilde + eta, eta ~ N(0, Q):
%              P = G'*inv(Q)*G, sparse and banded, and b = G\btilde
%   ll       : log p(y), with alpha integrated out
%   alphahat : Tm x 1 posterior mean of alpha
%   K        : Tm x Tm posterior precision of alpha, P + Z'*iR*Z
%
% The value uses the identity p(y) = p(y|alpha)p(alpha)/p(alpha|y), evaluated at
% alpha = alphahat, where the exponent of the denominator is zero. The two
% quadratic forms left are sums of squared residuals, so no large terms cancel.
%
% Written for this toolkit.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.2 and Exercise 9.3.
% Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
% Likelihood Estimation in State Space Models, International Journal of
% Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120, Section 2.3.

function [ll, alphahat, K] = intlike(y, Z, iR, P, b)
Tn = length(y);
if ~isequal(size(iR), [Tn Tn])
    error('ssm:intlike:badR', 'iR must be %d x %d, the size of y', Tn, Tn);
end
K = P + Z'*iR*Z;
CK = chol(K,'lower');
alphahat = CK'\(CK\(P*b + Z'*(iR*y)));
r = y - Z*alphahat;
d = alphahat - b;
ll = -Tn/2*log(2*pi) + sum(log(diag(chol(iR,'lower')))) ...
    + sum(log(diag(chol(P,'lower')))) - sum(log(diag(CK))) ...
    - .5*(r'*iR*r + d'*P*d);
end
