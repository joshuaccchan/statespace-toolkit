% This script evaluates the likelihood of the GARCH-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [llike sig2] = loglike_garch_t(e,gam,sig20,nu)
T = length(e);
e2 = e.^2;
expgam = exp(gam);
sig2 = zeros(T,1);
sig2(1) = expgam(1) + expgam(3)*sig20;
for t=2:T
    sig2(t) = expgam(1) + expgam(2)*e2(t-1) + expgam(3)*sig2(t-1);
end
llike = -T/2*log(nu*pi) + T*gammaln((nu+1)/2) - T*gammaln(nu/2) ...
    -.5*sum(log(sig2)) - (nu+1)/2*sum(log(1+ e2./sig2/nu));
end 