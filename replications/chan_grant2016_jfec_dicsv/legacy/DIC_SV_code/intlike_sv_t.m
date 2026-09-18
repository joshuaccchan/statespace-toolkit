% This script estimates the log integrated likelihood of the SV-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = intlike_sv_t(s2,muh,phih,omegah2,nu,ht,M)

% ht is the starting value for the Newton-Raphson

T = length(s2);
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
HiSHdeltah = HiSH*deltah;
errh_out = 1; 
while errh_out> 10^(-3);
        % E-step    
    Eilam = (nu+1)./(nu+s2./exp(ht));
    s2Eilam = s2.*Eilam;
        % M-step
    htt = ht;
    errh_in = 1; 
    while errh_in> 10^(-3);
        eht = exp(htt);
        sieht = s2Eilam./eht; 
        fh = -.5 + .5*sieht;
        Gh = .5*sieht;
        Kh = HiSH + sparse(1:T,1:T,Gh);
        newht = Kh\(fh+Gh.*htt+HiSHdeltah);
        errh_in = max(abs(newht-htt));
        htt = newht;          
    end    
    errh_out = max(abs(ht-htt));
    ht = htt;
end
    % compute negative Hessian
Gh = (nu+1)/(2*nu)*(s2.*exp(ht))./((exp(ht)+s2/nu).^2);
Kh = HiSH + sparse(1:T,1:T,Gh);
CKh = chol(Kh,'lower');

c = -.5*(T*log(omegah2) - log(1-phih^2)) - sum(log(diag(CKh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + CKh'\randn(T,M);
for i=1:M    
    h = bigh(:,i);    
    store_llike(i) = c -.5*(h-deltah)'*HiSH*(h-deltah) + deny_h(s2,h,nu) + ...
        + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
end

function llike = deny_h(s2,h,nu)
T = size(s2,1);
c = T*(gammaln((nu+1)/2) - gammaln(nu/2) - .5*log(nu*pi));
llike = c - .5*sum(h) -(nu+1)/2*sum(log(1+(s2./exp(h))/nu));
end
