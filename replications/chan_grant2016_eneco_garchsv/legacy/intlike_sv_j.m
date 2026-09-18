% This script estimates the integrated likelihood of the SV-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.
function llike = intlike_sv_j(y,mu,kappa,delta,muh,phih,omegah2,h0,M)

% h0 is the starting value for the Newton-Raphson

muk = delta(1);
sigk2 = exp(delta(2));
T = length(y);
c = -T/2*log(2*pi) -.5*(T*log(omegah2) - log(1-phih^2));
store_llike = zeros(M,1);
p1 = kappa*normpdf(y,mu+muk,sqrt(sigk2+exp(h0)));
p0 = (1-kappa)*normpdf(y,mu,exp(h0/2));
qprob = min(max(p1./(p0+p1),.001),.999);
lqprob = log(qprob);
lqprob0 = log(1-qprob);
bigq = binornd(1,repmat(qprob,1,M));    
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HiSH = Hphi'*sparse(1:T,1:T,[(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)])*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
HiSHdeltah = HiSH*deltah;
k = zeros(T,1);
for i=1:M     
    q = bigq(:,i);    
    id0 = find(q==0);
    id1 = find(q==1);
    n0 = length(id0);    
    Dk1 = 1./(1/sigk2 + 1./exp(h0(id1)));
    k1hat = Dk1 .* (muk/sigk2 + (y(id1)-mu)./exp(h0(id1)));    
    k(id0) = muk + sqrt(sigk2)*randn(n0,1);    
    k(id1) = k1hat + sqrt(Dk1).*randn(T-n0,1);    
    s2 = (y-mu-q.*k).^2;
    ht = h0;    errh = 1; 
    while errh> 10^(-3);
        expht = exp(ht);
        sinvexpht = s2./expht;  
        fh = -.5 + .5*sinvexpht;
        Gh = .5*sinvexpht;
        Kh = HiSH + sparse(1:T,1:T,Gh);
        newht = Kh\(fh+Gh.*ht+HiSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;       
    end 
    cholHh = chol(Kh,'lower');
    h = ht + cholHh'\randn(T,1); 
    
    store_llike(i) = c - sum(log(diag(cholHh))) - .5*(h-deltah)'*HiSH*(h-deltah) ...
        - .5*sum(h) - .5*exp(-h)'*s2 + .5*(h-ht)'*Kh*(h-ht) ...
        + q'*(log(kappa)-lqprob) + (1-q)'*(log(1-kappa)-lqprob0) ...
        - (T-n0)/2*log(sigk2) - .5/sigk2*sum((k(id1)-muk).^2) ...
        + .5*sum(log(Dk1)) + sum(.5./Dk1.*(k(id1)-k1hat).^2);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
end