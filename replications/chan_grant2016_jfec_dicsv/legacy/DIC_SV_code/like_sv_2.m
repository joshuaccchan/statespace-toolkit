% This function evaluates the log complete-data likelihood of a SV model with
% an AR(2) log volatility process (SV-2).
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = like_sv_2(s2,h,muh,phih,rhoh,omegah2)
T = length(s2);
Hthetah = speye(T) - sparse(3:T,2:(T-1),phih*ones(1,T-2),T,T) ...
    - sparse(3:T,1:(T-2),rhoh*ones(1,T-2),T,T);
intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));
HinvSH = Hthetah'*spdiags([1/intvar;1/intvar;1/omegah2*ones(T-2,1)],0,T,T)*Hthetah;
deltah = Hthetah\[muh;muh;muh*(1-phih-rhoh)*ones(T-2,1)];
c = -T*log(2*pi) -.5*((T-2)*log(omegah2) + 2*log(intvar));
u = h-deltah;
llike = c - .5*u'*HinvSH*u  - .5*sum(h) - .5*exp(-h)'*s2;
end