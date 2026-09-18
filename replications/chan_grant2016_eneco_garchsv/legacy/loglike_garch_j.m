% This script evaluates the likelihood of the GARCH-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

function [llike sig2] = loglike_garch_j(e,delta,kappa,gam,sig20)
T = length(e);
e2 = e.^2;
muk = delta(1);
sigk2 = exp(delta(2));
expgam = exp(gam);
sig2 = zeros(T,1);
sig2(1) = expgam(1) + expgam(3)*sig20;
for t=2:T    
    sig2(t) = expgam(1) + expgam(2)*e2(t-1) + expgam(3)*sig2(t-1);
end
llike0 = -.5*log(2*pi*sig2) -.5*e2./sig2;
llike1 = -.5*log(2*pi*(sig2+sigk2)) -.5*(e-muk).^2./(sig2+sigk2);
llike = sum(log(kappa*exp(llike1) + (1-kappa)*exp(llike0)));
end 