% This script computes the forecasts from the standard SV model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'SV';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute a few things
muh = log(var(yt)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(Tt,1);
exph = exp(h);
Hphi = speye(Tt) - sparse(2:Tt,1:(Tt-1),phih*ones(1,Tt-1),Tt,Tt);
newnuh = Tt/2 + nuh;

for isim = 1:nsims + burnin  
        % sample mu    
    invDmu = 1/Vmu + sum(1./exph);
    muhat = invDmu\sum(yt./exph);   
    mu = muhat + chol(invDmu,'lower')'\randn;           
        % sample h    
    HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(Tt-1,1)],0,Tt,Tt)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(Tt-1,1)];
    HinvSHdeltah = HinvSH*deltah;
    s2 = (yt-mu).^2;
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
    if alpMH > log(rand) || isim < 10 
        h = hc;
        exph = exp(h);
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
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;    
    
    if isim > burnin
        isave = isim - burnin;
        htp1 = muh + phih*(h(end)-muh) + sqrt(omegah2)*randn;
        sig2tp1 = exp(htp1);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-mu).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

