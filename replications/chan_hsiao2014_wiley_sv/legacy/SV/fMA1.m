% % =======================================================================
% % log-likelihood for MA(1) stochastic volatility model
% %
% % See Chan, J.C.C. and Hsiao, C.Y.L (2014). Estimation of Stochastic
% % Volatility Models with Heavy Tails and Serial Dependence. 
% % In: I. Jeliazkov and X.S. Yang (Eds.), Bayesian Inference in the 
% % Social Sciences, 159-180, John Wiley & Sons, New York.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================
function ell = fMA1(psi,v,h)
    T = length(v);
    Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
    Sigy = Hpsi*sparse(1:T,1:T,exp(h))*Hpsi';
    ell = -T/2*log(2*pi) -.5*sum(h) - .5*v'*(Sigy\v);
end