% This script evaluates the log integrated likelihood of the HP-AR model
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

function llike = intlike_HP_aug(y,phi,sigc2,tau0,lam)
T = length(y);
H2 = speye(T) - 2*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
    + sparse(3:T,1:(T-2),ones(1,T-2),T,T);
H2H2 = H2'*H2;
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
    - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);

HphiHphi = Hphi'*Hphi;
alp_tau = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];    
Ktau = (lam*H2H2 + HphiHphi)/sigc2;    
dtau = (lam*H2H2*alp_tau + HphiHphi*y)/sigc2;    
llike = -T/2*log(2*pi*sigc2^2/lam) - sum(log(diag(chol(Ktau))))...
    - .5*(y'*HphiHphi*y/sigc2 + lam/sigc2*alp_tau'*H2H2*alp_tau - dtau'*(Ktau\dtau)); 
end