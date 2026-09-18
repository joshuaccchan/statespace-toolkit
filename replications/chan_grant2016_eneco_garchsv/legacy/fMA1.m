%% negative log-likelihood for the MA(1) model
function ell = fMA1(psi,y,h)
T = length(y);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
L = Hpsi*sparse(1:T,1:T,exp(h))*Hpsi';
ell = -T/2*log(2*pi) -.5*sum(h) - .5*y'*(L\y);
ell = -ell;
end