% This script computes the forecasts from the SV-MA model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'SV-MA';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute a few things
mu = mean(yt);
muh = log(var(yt)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(Tt,1);    
newnuh = Tt/2 + nuh;
f = @(x) fMA1(x,yt-mu,h) + .5*(psi0-x)^2/Vpsi;  % negative of the log-density
psi = fminbnd(f,-.98,.98);
Hpsi = speye(Tt) + sparse(2:Tt,1:(Tt-1),psi*ones(1,Tt-1),Tt,Tt); 
Hphi = speye(Tt) - sparse(2:Tt,1:(Tt-1),phih*ones(1,Tt-1),Tt,Tt);
X = ones(Tt,1);
counth = 0; countpsi = 0; countphih = 0;
for isim = 1:nsims + burnin  
        % sample mu    
    Xtilde = Hpsi\X;    ytilde = Hpsi\yt;
    temp1 = Xtilde'*sparse(1:Tt,1:Tt,exp(-h));
    Dmu = 1/(temp1*Xtilde + 1/Vmu);
    muhat = Dmu*(mu0/Vmu + temp1*ytilde);
    mu = muhat + sqrt(Dmu)*randn;    
        % sample h    
    HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(Tt-1,1)],0,Tt,Tt)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(Tt-1,1)];
    HinvSHdeltah = HinvSH*deltah;
    s2 = (Hpsi\(yt-mu)).^2;
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
    end   
        % sample omegah2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];
    newSh = Sh + sum(errh.^2)/2;
    omegah2 = 1/gamrnd(newnuh, 1./newSh);    
        % sample phih
    Xphih = h(1:end-1)-muh;
    yphih = h(2:end) - muh;
    Dphih = 1/(1/Vphih + Xphih'*Xphih/omegah2);
    phihhat = Dphih*(phih0/Vphih + Xphih'*yphih/omegah2);
    phihc = phihhat + sqrt(Dphih)*randn;
    g = @(x) -.5*log(omegah2./(1-x.^2))-.5*(1-x.^2)/omegah2*(h(1)-muh)^2;
    if abs(phihc)<.9999
        alpMH = exp(g(phihc)-g(phih));
        if alpMH>rand
            phih = phihc;
            Hphi = speye(Tt) - sparse(2:Tt,1:(Tt-1),phih*ones(1,Tt-1),Tt,Tt);            
        end
    end    
        % sample muh    
    Dmuh = 1/(1/Vmuh + ((Tt-1)*(1-phih)^2 + (1-phih^2))/omegah2);
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;    
        % sample psi
    f = @(x) fMA1(x,yt-mu,h) + .5*(psi0-x)^2/Vpsi;  % negative of the log-density
    psihat = fminbnd(f,-.98,.98);
    sqVpsic = .04; Vpsic = sqVpsic^2;
    psic = psihat + sqVpsic*randn;
    if abs(psic)<.99
        alpMH = -f(psic) + f(psi) ...
            -.5*(psi-psihat)^2/Vpsic + .5*(psic-psihat)^2/Vpsic;
    else
        alpMH = -inf;
    end
     if alpMH>log(rand)
        psi = psic;
        Hpsi = speye(Tt) + sparse(2:Tt,1:(Tt-1),psi*ones(1,Tt-1),Tt,Tt);         
    end 
    
    if isim > burnin
        isave = isim - burnin;
        u = Hpsi\(yt-mu);
        ym = mu + psi*u(end);
        htp1 = muh + phih*(h(end)-muh) + sqrt(omegah2)*randn;
        sig2tp1 = exp(htp1);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-ym).^2/sig2tp1; 
        tempyhat1(isave,:) = [ym lden]; 
    end
end

