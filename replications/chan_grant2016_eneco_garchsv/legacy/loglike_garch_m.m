% This script evaluates the likelihood of the GARCH-M model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [llike sig2] = loglike_garch_m(e,gam,lam,sig20)
T = length(e);
expgam = exp(gam);
sig2 = zeros(T,1);
sig2(1) = expgam(1) + expgam(3)*sig20;
for t=2:T
    et = e(t-1) - lam*sig2(t-1);
    sig2(t) = expgam(1) + expgam(2)*et^2 + expgam(3)*sig2(t-1);
end
u = e-lam*sig2;
llike = -T/2*log(2*pi) -.5*sum(log(sig2)) -.5*sum(u.^2./sig2);
if isnan(llike)
    llike = -10^100;
end
end 