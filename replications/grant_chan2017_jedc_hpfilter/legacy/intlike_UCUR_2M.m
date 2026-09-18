% This script evaluates the log integrated likelihood of the UCUR-2M model
% 
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

function llike = intlike_UCUR_2M(y,phi,sigy2,sigtau2,rho,tau0)
T = length(y);
H2 = speye(T) - 2*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
    + sparse(3:T,1:(T-2),ones(1,T-2),T,T);
H2H2 = H2'*H2;
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
    - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
alp = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];
a = -rho*sqrt(sigy2/sigtau2)*(H2*alp);
B = Hphi + rho*sqrt(sigy2/sigtau2)*H2;
tmpc = 1/((1-rho^2)*sigy2);
Ktau = H2H2/sigtau2 + tmpc*(B'*B);
dtau = H2H2*alp/sigtau2 + tmpc*B'*(Hphi*y-a);
llike = -T/2*log(2*pi*(1-rho^2)*sigy2*sigtau2) - sum(log(diag(chol(Ktau))))...
    - .5*(tmpc*(Hphi*y-a)'*(Hphi*y-a) + 1/sigtau2*alp'*H2H2*alp - dtau'*(Ktau\dtau)); 
end