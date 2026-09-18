% this function evaluates the log integrated likelihood of the SV model
% with MA(1) errors (SV-MA).
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

% ht is the starting value for the Newton-Raphson
% M is the # of draws for the importance sampling
function llike = intlike_sv_ma(e,psi,muh,phih,omegah2,ht,M)
T = length(e);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
HiSHdeltah = HiSH*deltah;
s2 = (Hpsi\e).^2;
errh = 1; 
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
c = -T/2*log(2*pi) -.5*(T*log(omegah2) - log(1-phih^2)) - sum(log(diag(cholHh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + cholHh'\randn(T,M);
for i=1:M    
    h = bigh(:,i);    
    L = Hpsi*sparse(1:T,1:T,exp(h))*Hpsi';
    store_llike(i) = c - .5*(h-deltah)'*HiSH*(h-deltah) - .5*sum(h) - .5*e'*(L\e) + ...
        + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
end

