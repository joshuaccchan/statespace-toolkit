% This script computes the forecasts from the SV-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'SV-t';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute a few things
nu = 5;
lam = 1./gamrnd(nu/2,2/nu,Tt,1);
muh = log(var(yt)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(Tt,1);
exph = exp(h);    
Hphi = speye(Tt) - sparse(2:Tt,1:(Tt-1),phih*ones(1,Tt-1),Tt,Tt);
newnuh = Tt/2 + nuh;

for isim = 1:nsims + burnin  
        % sample mu    
    invDmu = 1/Vmu + sum(1./(exph.*lam));
    muhat = invDmu\sum(yt./(exph.*lam));   
    mu = muhat + chol(invDmu,'lower')'\randn; 
        % sample h    
    HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(Tt-1,1)],0,Tt,Tt)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(Tt-1,1)];
    HinvSHdeltah = HinvSH*deltah;
    s2 = (yt-mu).^2./lam;
    errh = 1; ht = h;
    while errh> 10^(-3)
        expht = exp(ht);
        sinvexpht = s2./expht;        
        fh = -.5 + .5*sinvexpht;
        Gh = .5*sinvexpht;
        Kh = HinvSH + spdiags(Gh,0,Tt,Tt);
        newht = Kh\(fh+Gh.*ht+HinvSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Kh,'lower');
    % AR-step:     
    hstar = ht;
    uh = hstar-deltah;
    logc = -.5*uh'*HinvSH*uh -.5*sum(hstar) - .5*exp(-hstar)'*s2 + log(3);
    flag = 0;
    while flag == 0
        hc = ht + cholHh'\randn(Tt,1);
        vhc = hc-ht;
        uhc = hc-deltah;
        alpARc = -.5*uhc'*HinvSH*uhc -.5*sum(hc) + ...
            -.5*exp(-hc)'*s2 + .5*vhc'*Kh*vhc - logc;            
        if alpARc > log(rand)
            flag = 1;
        end
    end        
    % MH-step
    vh = h-ht;
    uh = h-deltah;
    alpAR = -.5*uh'*HinvSH*uh -.5*sum(h) + ...
        -.5*exp(-h)'*s2 + .5*vh'*Kh*vh - logc;
    if alpAR < 0
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end    
    if alpMH > log(rand) || isim == 1
        h = hc;
        exph = exp(h);        
    end 
        % sample lam
    lam = 1./gamrnd((nu+1)/2,2./(nu+(yt-mu).^2./exph));
        % sample nu                
    sum1 = sum(log(lam));
    sum2 = sum(1./lam);
    fnu = @(x) Tt*(x/2.*log(x/2)-gammaln(x/2)) - (x/2+1)*sum1 - x/2*sum2;
    dfnu = @(x) Tt/2*(log(x/2)+1-psi(x/2)) - .5*(sum1+sum2);  % score
    d2fnu = @(x) Tt/(2*x) - Tt/4*psi(1,x/2);                   % Hessian
    err_nu = 1;
    nut = nu;
    while abs(err_nu) > 10^(-3) % stopping criteria
        err_nu = dfnu(nut);
        Knu = -d2fnu(nut);      % infomation matrix
        nut = nut + Knu\err_nu;
        if nut<2
            nut = 5;
            Knu = -f2(nut);
            break;
        end
    end
    sqrtDnu = sqrt(1/Knu);
    nuc = nut + sqrtDnu*randn; 
    if nuc>2 && nuc < nuub
        alpMH = exp(fnu(nuc)-fnu(nu)) ... 
            * normpdf(nu,nut,sqrtDnu)/normpdf(nuc,nut,sqrtDnu);
        if alpMH > rand
            nu = nuc;            
        end    
    end        
        % sample omegah2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];    
    newSh = Sh + sum(errh.^2)/2;    
    omegah2 = 1/gamrnd(newnuh, 1./newSh);    
        % sample phih
    Xphi = h(1:end-1)-muh;
    yphi = h(2:end) - muh;
    Dphi = 1/(1/Vphih + Xphi'*Xphi/omegah2);
    phihat = Dphi*(phih0/Vphih + Xphi'*yphi/omegah2);
    phic = phihat + sqrt(Dphi)*randn;
    g = @(x) -.5*log(omegah2./(1-x.^2))-.5*(1-x.^2)/omegah2*(h(1)-muh)^2;
    if abs(phic)<.9999
        alpMH = exp(g(phic)-g(phih));
        if alpMH>rand
            phih = phic;            
            Hphi = speye(Tt) - sparse(2:Tt,1:(Tt-1),phih*ones(1,Tt-1),Tt,Tt);
        end
    end    
        % sample muh    
    Dmuh = 1/(1/Vmuh + ((Tt-1)*(1-phih)^2 + (1-phih^2))/omegah2);
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + ...
        (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;   
    
    if isim > burnin
        isave = isim - burnin;
        htp1 = muh + phih*(h(end)-muh) + sqrt(omegah2)*randn;
        sig2tp1 = exp(htp1);        
        lden = gammaln((nu+1)/2) - gammaln(nu/2) - .5*log(nu*pi*sig2tp1) ...
            -(nu+1)/2*log(1+(y(t+1)-mu).^2/(nu*sig2tp1)); 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

