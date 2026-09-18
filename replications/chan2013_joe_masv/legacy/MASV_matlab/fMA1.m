% =======================================================================
% This script evaluates the negative log-likelihood of the UC-MA model.
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models 
% with Application to Inflation Forecast, Journal of Econometrics, 
% 176 (2), 162-172.
% =======================================================================

function ell = fMA1(psi,y,h)
T = length(y);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
Omega_y = Hpsi*sparse(1:T,1:T,exp(h))*Hpsi';
ell = -T/2*log(2*pi) -.5*sum(h) - .5*y'*(Omega_y\y);
ell = -ell;
end