% this function evaluates the log complete-data likelihood of the SV-L
% flag = 1: includes all consants; flag = 0: exclude terms not involving h
function llike = like_sv_l(y,h,mu,rho,muh,phih,omegah2,flag)
T = length(h)-1;    
Hphi = speye(T+1) - phih*sparse(2:T+1,1:T,ones(1,T),T+1,T+1);    
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T,1)],0,T+1,T+1)*Hphi;
deltah = Hphi\[muh;(1-phih)*muh*ones(T,1)]; 
tmp1 = (y-mu)./exp(h(1:T)/2);
eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;

if flag == 1
    c = -T/2*log(2*pi*(1-rho^2)) - T/2*log(2*pi) -.5*((T+1)*log(omegah2) ...
        - log(1-phih^2));
elseif flag == 0
    c = 0;
end
llike = c -.5*(h-deltah)'*HiSH*(h-deltah) ...
    -.5*sum(h(1:T)) - .5/(1-rho.^2)*sum((tmp1-rho/sqrt(omegah2)*eh).^2);
    
end