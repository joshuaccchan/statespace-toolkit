% This script evaluates the log complete-data likelihood of the standard SV
% model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = like_sv(s2,h,muh,phih,omegah2)
T = length(s2);
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
c = -T*log(2*pi) -.5*(T*log(omegah2) - log(1-phih^2));
u = h-deltah;
llike = c - .5*u'*HinvSH*u  - .5*sum(h) - .5*exp(-h)'*s2;
end