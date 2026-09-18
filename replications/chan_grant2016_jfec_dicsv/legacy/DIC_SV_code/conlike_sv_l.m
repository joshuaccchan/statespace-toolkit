% This function evaluates the log conditional likelihood of the SV-L model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.
function llike = conlike_sv_l(y,h,mu,rho,muh,phih,omegah2)
T = length(h)-1;    
tmp1 = (y-mu)./exp(h(1:T)/2);
eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;
llike = -T/2*log(2*pi*(1-rho^2)) -.5*sum(h(1:T)) ...
    - .5/(1-rho.^2)*sum((tmp1-rho/sqrt(omegah2)*eh).^2);    
end
