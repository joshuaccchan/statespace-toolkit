% This script evaluates the log conditional likelihood of the SV-M model.
% See:
% 
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = conlike_sv_m(s2,h)
T = length(s2);
llike = -T/2*log(2*pi) - .5*sum(h) - .5*exp(-h)'*s2;
end