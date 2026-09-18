% This function evaluates the log complete-data likelihood of the SV-t
% model
% See:
% 
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = like_sv_t(s2,h,lam,muh,phih,omegah2,nu)
T = length(s2);
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
c = -T*log(2*pi) -.5*(T*log(omegah2) - log(1-phih^2)) + .5*T*nu*log(nu/2) - T*gammaln(nu/2);

llike = c -.5*(h-deltah)'*HinvSH*(h-deltah) -.5*sum(log(lam)) ...
    -.5*sum(h) -.5*(exp(-h)./lam)'*s2 -(nu/2+1)*sum(log(lam)) - .5*sum(nu./lam);
end