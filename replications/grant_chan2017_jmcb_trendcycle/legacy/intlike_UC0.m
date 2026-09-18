function llike = intlike_UC0(y,mu,phi,sigc2,sigtau2,tau0)
rho = 0;
T = length(y);
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
    - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
alp = H\(mu+[tau0;sparse(T-1,1)]);
tmpc = 1/((1-rho^2)*sigc2);
a = -rho*sqrt(sigc2/sigtau2)*(H*alp);
B = Hphi + rho*sqrt(sigc2/sigtau2)*H;
Ktau = HH/sigtau2 + tmpc*(B'*B);
dtau = HH*alp/sigtau2 + tmpc*B'*(Hphi*y-a);
llike = -T/2*log(2*pi*(1-rho^2)*sigc2*sigtau2) - sum(log(diag(chol(Ktau))))...
    - .5*(tmpc*(Hphi*y-a)'*(Hphi*y-a) + 1/sigtau2*alp'*HH*alp - dtau'*(Ktau\dtau));   
end