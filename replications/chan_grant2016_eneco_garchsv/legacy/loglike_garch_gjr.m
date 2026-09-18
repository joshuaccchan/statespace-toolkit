% This script evaluates the likelihood of the GARCH-GJR model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [llike sig2] = loglike_garch_gjr(e,gam,del,sig20)
T = length(e);
e2 = e.^2;
expgam = exp(gam);
sig2 = zeros(T,1);
sig2(1) = expgam(1) + expgam(3)*sig20;
for t=2:T
    sig2(t) = expgam(1) + (expgam(2) + del*(e(t-1)<0))*e2(t-1) + expgam(3)*sig2(t-1);
end
if sig2(end)>10^10 || sum(sig2<0) > 0
    llike = -10^(100);
else
    llike = -T/2*log(2*pi) -.5*sum(log(sig2)) -.5*sum(e2./sig2);
end
end 