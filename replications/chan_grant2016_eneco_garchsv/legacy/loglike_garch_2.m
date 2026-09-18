% This script evaluates the likelihood of the standard GARCH-2 model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [llike sig2] = loglike_garch_2(e,gam,sig20)
T = length(e);
e2 = e.^2;
expgam = exp(gam);
sig2 = zeros(T,1);
sig2(1) = expgam(1) + expgam(3)*sig20;
sig2(2) = expgam(1) + expgam(2)*e2(1) + expgam(3)*sig2(1) + expgam(4)*sig20;
for t=3:T
    sig2(t) = expgam(1) + expgam(2)*e2(t-1) + expgam(3)*sig2(t-1) ...
        + expgam(4)*sig2(t-2);
end
if sum(isnan(sig2)) == 0    
    llike = -T/2*log(2*pi) -.5*sum(log(sig2)) -.5*sum(e2./sig2);
else
    llike = -10^100;
end
end 