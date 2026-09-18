% This script computes the forecasts from a SV model with an AR(2) log 
% volatility process (SV-2)
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'SV-2';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]

    % initialize the Markov chain
muh = log(var(yt)); 
phih = .95; 
rhoh = 0;
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(Tt,1);
exph = exp(h);
    % compute a few things
Hthetah = speye(Tt) - sparse(3:Tt,2:(Tt-1),phih*ones(1,Tt-2),Tt,Tt) ...
    - sparse(3:Tt,1:(Tt-2),rhoh*ones(1,Tt-2),Tt,Tt);
newnuh = Tt/2 + nuh;
for isim = 1:nsims + burnin  
        % sample mu    
    invDmu = 1/Vmu + sum(1./exph);
    muhat = invDmu\sum(yt./exph);   
    mu = muhat + chol(invDmu,'lower')'\randn;           
        % sample h    
    intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));
    HinvSH = Hthetah'*spdiags([1/intvar;1/intvar;1/omegah2*ones(Tt-2,1)],0,Tt,Tt)*Hthetah;
    deltah = Hthetah\[muh;muh;muh*(1-phih-rhoh)*ones(Tt-2,1)];
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
    logc = -.5*uh'*HinvSH*uh -.5*sum(hstar) - .5*exp(-hstar)'*s2 + log(4);
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
        % sample omegah2
    temp1 = (1-rhoh)/((1+rhoh)*((1-rhoh)^2-phih^2));
    errh = [(h(1)-muh)/sqrt(temp1); (h(2)-muh)/sqrt(temp1); ...
        h(3:end)-phih*h(2:end-1)-rhoh*h(1:end-2)-muh*(1-phih-rhoh)];    
    newSh = Sh + sum(errh.^2)/2;    
    omegah2 = 1/gamrnd(newnuh, 1./newSh);    
    
        % sample phih and rhoh
    Xthetah = [h(2:end-1)-muh h(1:end-2)-muh];
    ythetah = h(3:end) - muh;
    invDthetah = diag([1/Vphih 1/Vrhoh]) + Xthetah'*Xthetah/omegah2;
    thetahhat = invDthetah\([phih0/Vphih;rhoh0/Vrhoh] + Xthetah'*ythetah/omegah2);
    thetahc = thetahhat + chol(invDthetah,'lower')\randn(2,1);        
    g = @(x) -log((1-x(2))*omegah2/((1+x(2))*((1-x(2))^2-x(1)^2))) ...
        -.5/((1-x(2))*omegah2/((1+x(2))*((1-x(2))^2-x(1)^2)))*(h(1)+h(2)-2*muh)^2;    
    if sum(thetahc)< .999 && thetahc(2)-thetahc(1)<.999 && abs(thetahc(2))<.999
        alpMH = exp(g(thetahc)-g([phih rhoh]));
        if alpMH>rand
            phih = thetahc(1);
            rhoh = thetahc(2);            
            Hthetah = speye(Tt) - sparse(3:Tt,2:(Tt-1),phih*ones(1,Tt-2),Tt,Tt) ...
                - sparse(3:Tt,1:(Tt-2),rhoh*ones(1,Tt-2),Tt,Tt);
        end
    end
    
    	% sample muh
    intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));
    Dmuh = 1/(1/Vmuh + (Tt-2)*(1-phih-rhoh)^2/omegah2 + 2/intvar);
    muhhat = Dmuh*(muh0/Vmuh + (h(1)+h(2))/intvar ...
        + (1-phih-rhoh)/omegah2*sum(h(3:end)-phih*h(2:end-1)-rhoh*h(1:end-2)));
    muh = muhhat + sqrt(Dmuh)*randn;   
    
    if isim > burnin
        isave = isim - burnin;
        htp1 = muh + phih*(h(end)-muh) + rhoh*(h(end-1)-muh) + sqrt(omegah2)*randn;
        sig2tp1 = exp(htp1);        
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-mu).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

