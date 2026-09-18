% This function estimates the log integrated likelihood of a SV model with
% an AR(2) log volatility process (SV-2).
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

function llike = intlike_sv_2(s2,muh,phih,rhoh,omegah2,ht,M)
T = length(s2);
Hthetah = speye(T) - sparse(3:T,2:(T-1),phih*ones(1,T-2),T,T) ...
    - sparse(3:T,1:(T-2),rhoh*ones(1,T-2),T,T);
intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));
HinvSH = Hthetah'*spdiags([1/intvar;1/intvar;1/omegah2*ones(T-2,1)],0,T,T)*Hthetah;
deltah = Hthetah\[muh;muh;muh*(1-phih-rhoh)*ones(T-2,1)];
HinvSHdeltah = HinvSH*deltah;
errh = 1; 
while errh> 10^(-3);
    expht = exp(ht);
    sinvexpht = s2./expht;        
    fh = -.5 + .5*sinvexpht;
    Gh = .5*sinvexpht;
    Kh = HinvSH + spdiags(Gh,0,T,T);
    newht = Kh\(fh+Gh.*ht+HinvSHdeltah);
    errh = max(abs(newht-ht));
    ht = newht;          
end 
cholHh = chol(Kh,'lower');
c = -T/2*log(2*pi) -.5*((T-2)*log(omegah2)+2*log(intvar)) - sum(log(diag(cholHh)));
store_llike = zeros(M,1);
bigh = repmat(ht,1,M) + cholHh'\randn(T,M);
for i=1:M    
    h = bigh(:,i);    
    store_llike(i) = c - .5*(h-deltah)'*HinvSH*(h-deltah) - .5*sum(h) ...
        - .5*exp(-h)'*s2 + .5*(h-ht)'*Kh*(h-ht);
end
maxllike = max(store_llike);
llike = maxllike + log(mean(exp(store_llike-maxllike)));
end

