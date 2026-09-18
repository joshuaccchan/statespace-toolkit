% This function evaluates the log conditional likelihood of the SV model
% with MA(1) errors (SV-MA).
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.
function llike = conlike_sv_ma(e,h,psi)
T = length(e);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
L = Hpsi*sparse(1:T,1:T,exp(h))*Hpsi';
llike = -T/2*log(2*pi) -.5*sum(h) - .5*e'*(L\e);
end