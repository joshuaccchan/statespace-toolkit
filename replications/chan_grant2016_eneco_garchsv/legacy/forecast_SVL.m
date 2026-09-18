% This script computes the forecasts from the SV-L model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'SV-L';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute a few things
lrhopri = @(x) -.5*(x-rho0).^2/Vrho;
mu = mean(yt);
phih = .98;
omegah2 = .01;
rho = 0;
muh = -10;
Hphi = speye(Tt+1) - phih*sparse(2:Tt+1,1:Tt,ones(1,Tt),Tt+1,Tt+1);
HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(Tt,1)],0,Tt+1,Tt+1)*Hphi;
deltah = Hphi\[muh;(1-phih)*muh*ones(Tt,1)]; 
h = deltah + chol(HinvSH,'lower')'\randn(Tt+1,1);    
for isim = 1:nsims + burnin  
        % sample mu
    eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;
    iexph = exp(-h(1:Tt));
    Dmu = 1/(1/Vmu + 1/(1-rho^2)*sum(iexph));
    muhat = Dmu*(mu0/Vmu + ...
        1/(1-rho^2)*iexph'*(yt-rho/sqrt(omegah2)*exp(h(1:Tt)/2).*eh));    
    mu = muhat + sqrt(Dmu)*randn;         
        % sample h  
    HinvSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(Tt,1)],0,Tt+1,Tt+1)*Hphi;
    deltah = Hphi\[muh;(1-phih)*muh*ones(Tt,1)]; 
    errh = 1; ht = h;
    c1 = 1/(1-rho^2);
    c2 = rho/sqrt(omegah2);
    c3 = c2^2;        
    while errh > 10^(-3)
        tmp1 = (yt-mu)./exp(ht(1:Tt)/2);                
        tmp2 = tmp1.^2;        
        eh = ht(2:end)-phih*ht(1:end-1)-(1-phih)*muh;
        fh1 = -.5 - c1/2*(-tmp2 -2*c3*phih*eh + c2*tmp1.*(eh+2*phih));        
        fh2 = c1*c2*(tmp1 - c2*eh);        
        fh = [fh1;0] + [0;fh2];    
        Gh1 = c1/2*(tmp2 + 2*c3*phih^2 - c2/2*tmp1.*(eh + 4*phih));
        Gh2 = c1*c3*ones(Tt,1);
        Gh3 = -c1*c2*(c2*phih - .5*tmp1);
        Gh = sparse(1:Tt+1,1:Tt+1,[Gh1;0]+[0;Gh2]) ...
            + sparse(2:Tt+1,1:Tt,Gh3,Tt+1,Tt+1) + sparse(1:Tt,2:Tt+1,Gh3,Tt+1,Tt+1);      
        Kh = HinvSH + Gh;
        newht = Kh\(HinvSH*deltah + fh + Gh*ht);
        errh = max(abs(newht-ht));
        ht = newht;
    end    
    % AR-step
    logc = like_sv_l(yt,ht,mu,rho,muh,phih,omegah2,0) + log(3);
    CinvDh = chol(Kh,'lower');
    flag = 0;
    while flag == 0
        hc = ht + CinvDh'\randn(Tt+1,1);      
        alpARc = like_sv_l(yt,hc,mu,rho,muh,phih,omegah2,0) + ...
            + .5*(hc-ht)'*Kh*(hc-ht) - logc;
        if alpARc > log(rand)
            flag = 1;
        end
    end       
    % MH-step    
    alpAR = like_sv_l(yt,h,mu,rho,muh,phih,omegah2,0) + ...
        .5*(h-ht)'*Kh*(h-ht) - logc;
    if alpAR < 0 
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end        
    if alpMH > log(rand) || isim<10
        h = hc;
    end    
        % sample rho - griddy Gibbs
    eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;
    tmprho = exp(-h(1:Tt)/2).*(yt-mu);
    k1 = tmprho'*tmprho;
    k2 = tmprho'*eh;
    k3 = eh'*eh; 
    grho = @(x) lrhopri(x) -Tt/2*log(1-x.^2) + ...
        - (k1 - 2*x/sqrt(omegah2)*k2 + x.^2/omegah2*k3)./(2*(1-x.^2));    
    rhogrid = linspace(-1+rand/100,1-rand/100,300)';
    rhopdf = grho(rhogrid);
    rhopdf = exp(rhopdf-max(rhopdf));
    rhocdf = cumsum(rhopdf);
    rhocdf = rhocdf/rhocdf(end);    
    rho = rhogrid(find(rhocdf>rand,1)); 
        % sample omegah2
    eh = h(2:end)-phih*h(1:end-1)-(1-phih)*muh;
    eh_ss = eh'*eh;
    newSh = Sh + ((1-phih^2)*(h(1)-muh)^2 + eh_ss)/2;
    omegah2c = 1/gamrnd(nuh+(Tt+1)/2, 1/newSh);
    k1 = .5*rho^2/(1-rho^2)*eh_ss;
    k2 = rho/(1-rho^2)*eh'*((yt-mu)./exp(h(1:Tt)/2));
    goh2 = @(x) -k1/x + k2/sqrt(x);
    alpMH = goh2(omegah2c) - goh2(omegah2);
    if alpMH > log(rand)
        omegah2 = omegah2c;        
    end    
        % sample phih
    tmph = h-muh;
    omegah = sqrt(omegah2);
    Dphih = 1/(1/Vphih + sum(tmph(1:Tt).^2)/(omegah2*(1-rho^2)));
    phihhat = Dphih*(phih0/Vphih + tmph(1:Tt)'*tmph(2:Tt+1)/omegah2 ...
        - rho/((1-rho^2)*omegah)*tmph(1:Tt)'*((yt-mu)./exp(h(1:Tt)/2)-rho/omegah*tmph(2:Tt+1)));    
    phihc = phihhat + sqrt(Dphih)*randn;    
    gphih = @(x) -.5*log(omegah2./(1-x.^2)) -.5*(1-x.^2)/omegah2*(h(1)-muh)^2;
    if abs(phihc)<.999
        alpMH = gphih(phihc)-gphih(phih);
        if alpMH > log(rand)
            phih = phihc;            
        end
    end 
    Hphi = speye(Tt+1) - phih*sparse(2:Tt+1,1:Tt,ones(1,Tt),Tt+1,Tt+1);    
        % sample muh  
    tmph = h(2:end)-phih*h(1:end-1);
    omegah = sqrt(omegah2);
    Dmuh = 1/(1/Vmuh + Tt*(1-phih)^2/omegah2 + (1-phih^2)/omegah2 ...
        + Tt*(1-phih)^2*rho^2/(omegah2*(1-rho^2)));
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(tmph) ...
        - rho*(1-phih)/(omegah*(1-rho^2))*sum((yt-mu)./exp(h(1:Tt)/2) - rho/omegah*tmph));
    muh = muhhat + sqrt(Dmuh)*randn;
    
    if isim > burnin
        isave = isim - burnin;        
        htp1 = h(end);
        sig2tp1 = exp(htp1);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-mu).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

