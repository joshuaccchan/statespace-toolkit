% This script estimates the log integrated likelihood of the SV-L model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = intlike_sv_l(y,mu,rho,muh,phih,omegah2,ht,M)

% ht is the starting value for the Newton-Raphson

T = length(y);
Hphi = speye(T+1) - sparse(2:T+1,1:T,phih*ones(1,T),T+1,T+1);
HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T,1)],0,T+1,T+1)*Hphi;
deltah = Hphi\[muh;(1-phih)*muh*ones(T,1)]; 
errh = 1; 
c1 = 1/(1-rho^2);
c2 = rho/sqrt(omegah2);
c3 = c2^2;        
while errh > 10^(-3)
    tmp1 = (y-mu)./exp(ht(1:T)/2);                
    tmp2 = tmp1.^2;        
    eh = ht(2:end)-phih*ht(1:end-1)-(1-phih)*muh;
    fh1 = -.5 - c1/2*(-tmp2 -2*c3*phih*eh + c2*tmp1.*(eh+2*phih));        
    fh2 = c1*c2*(tmp1 - c2*eh);        
    fh = [fh1;0] + [0;fh2];    
    Gh1 = c1/2*(tmp2 + 2*c3*phih^2 - c2/2*tmp1.*(eh + 4*phih));
    Gh2 = c1*c3*ones(T,1);
    Gh3 = -c1*c2*(c2*phih - .5*tmp1);
    Gh = sparse(1:T+1,1:T+1,[Gh1;0]+[0;Gh2]) ...
        + sparse(2:T+1,1:T,Gh3,T+1,T+1) + sparse(1:T,2:T+1,Gh3,T+1,T+1);      
    Kh = HiSH + Gh;
    newht = Kh\(HiSH*deltah + fh + Gh*ht);
    errh = max(abs(newht-ht));
    ht = newht;
end    
cholHh = chol(Kh,'lower');
c = -T/2*log(2*pi*(1-rho^2)) -.5*((T+1)*log(omegah2) - log(1-phih^2)) ...
    - sum(log(diag(cholHh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + cholHh'\randn(T+1,M);
for i=1:M    
    h = bigh(:,i);    
    tmp1 = (y-mu)./exp(h(1:T)/2);
    eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;
    store_llike(i) = c - .5*(h-deltah)'*HiSH*(h-deltah) - .5*sum(h(1:T)) ...
        - .5/(1-rho.^2)*sum((tmp1-rho/sqrt(omegah2)*eh).^2) ...
        + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
end

