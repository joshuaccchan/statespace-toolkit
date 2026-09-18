% This script computes the forecasts from the Bi-UC model and its variants
% as described in Chan, Koop and Potter (2016)
% 
% See:
% Chan, J.C.C., Koop, G. and Potter, S.M. (2016). A Bounded Model of Time 
% Variation in Trend Inflation, NAIRU and the Phillips Curve, Journal of 
% Applied Econometrics, 31(3), 551-565.

tmpyhat1 = zeros(nsim,5);  %% [point forecast, prelike]
tmpyhat4 = zeros(nsim,5); 
tmpyhat8 = zeros(nsim,5); 
tmpyhat12 = zeros(nsim,5); 
tmpyhat16 = zeros(nsim,5);

% compute a few things outside the MCMC loop
id1 = (1:Tt)';
id3 = (1:Tt-1)';
Kh =  speye(Tt) - spdiags(ones(Tt-1,1),-1,Tt,Tt);
newnuu = Tt-2 + nuu0;
newnuh = Tt-1 + nuh0;
newnutaupi = Tt-1 + nutaupi0;
newnutauu = Tt-1 + nutauu0;
newnurhopi = Tt-1 + nurhopi0;
newnulam = Tt-1 + nulam0;

% initialize the Markov chain
sigh = .2;
sigu = .2;
sigtaupi = .02;
sigtauu = .01;
if tvp_rhopi == 1
    sigrhopi = .005;
elseif tvp_rhopi == 0
    sigrhopi = 1e-8;
end
if tvp_lam == 1
    siglam = .005;
elseif tvp_lam == 0    
    siglam = 1e-8;
end
h = ones(Tt,1);
rhopi = .4*ones(Tt,1);
rhou = [1.75 -.8]';
lam = -.4*ones(Tt,1);

Kpi = speye(Tt) - sparse(2:Tt,1:(Tt-1),rhopi(2:Tt),Tt,Tt);
mupi = Kpi\([rhopi(1)*(y0-taupi0); sparse(Tt-1,1)]);
expinvh = exp(-h);
alppi = Kh\[taupi0; sparse(Tt-1,1)];
Staupi = Kh'*sparse(id1,id1,[invVtaupi repmat(1/sigtaupi,1,Tt-1)],Tt,Tt)*Kh;
Sy = Kpi'*sparse(id1,id1, expinvh,Tt,Tt)*Kpi;
Hpi =  Sy + Staupi;
taut = Hpi\(Staupi*alppi + Sy*(yt-mupi));
taut = min(taut,bpi-.1);    taut = max(taut,api+.1);
count = 0; flag =0;
while flag == 0 && count<1000
    taupi = taut + chol(Hpi)\randn(Tt,1);
    if  max(taupi) <= bpi - .01 && min(taupi) >= api + .01
        flag = 1;
    end
    count = count + 1;
end
if count == 1000
    taupi = taut;
end

Ku = spdiags([ones(Tt,1) [-rhou(1)*ones(Tt-1,1); 0] [-rhou(2)*ones(Tt-2,1);0;0]], [0 -1 -2], Tt, Tt);
muu = Ku\[rhou(1)*(u0(2)-tauu0(2))+rhou(2)*(u0(1)-tauu0(1)); rhou(2)*(u0(2)-tauu0(2)); sparse(Tt-2,1)];
alpu = Kh\[tauu0(2); sparse(Tt-1,1)];
Su  = Ku'*sparse(id1,id1,[1/5 1/5 repmat(1/sigu,1,Tt-2)])*Ku;     %% u1 and u2 have variances 5
Stauu = Kh'*sparse(id1,id1,[invVtauu repmat(1/sigtauu,1,Tt-1)])*Kh;
Hu =  Su + Stauu;
taut = Hu\(Stauu*alpu + Su*(ut-muu));
taut = min(taut,bu-.1);    tauu = max(taut,au+.1);
count = 0; flag =0;
while flag == 0 && count<1000
    tauu = taut + chol(Hu)\randn(Tt,1);
    if  max(tauu) <= bu - .01 && min(tauu) >= au + .01
        flag = 1;
    end
    count = count + 1;
end
if count == 1000
    tauu = taut;
end
pistar = yt - taupi;
ustar = ut - tauu;

Sh = spdiags([invVh; 1/sigh*ones(Tt-1,1)],0,Tt,Tt);
KSKh = Kh'*Sh*Kh;
s2 = (pistar - rhopi.*[(y0-taupi0);pistar(id3)]).^2;
h = mean(log(s2)) + std(log(s2))*randn(Tt,1);
errh = 1; ht = h;
while errh> 10^(-3)
    expht = exp(ht);
    sexpht = s2./expht;
    fh = -.5 + .5*sexpht;
    Gh = .5*sexpht;
    Hh = KSKh + spdiags(Gh,0,Tt,Tt);
    newht = Hh\(fh+Gh.*ht);
    errh = max(abs(newht-ht));
    ht = newht;
end
h = ht + chol(Hh)\randn(Tt,1);

sqrtsigtaupi = sqrt(sigtaupi);
sqrtsigtauu = sqrt(sigtauu);
sqrtsigrhopi = sqrt(sigrhopi);
sqrtsiglam = sqrt(siglam);

% MCMC starts here
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
for isim = 1:nsim+burnin
    
        % sample taupi
    Kpi = speye(Tt) - sparse(2:Tt,1:(Tt-1),rhopi(2:Tt),Tt,Tt);
    mupi = Kpi\([rhopi(1)*(y0-taupi0); sparse(Tt-1,1)] + lam.*(ut-tauu)) ;
    expinvh = exp(-h);
    alppi = Kh\[taupi0; sparse(Tt-1,1)];
    Staupi = Kh'*sparse(id1,id1,[invVtaupi repmat(1/sigtaupi,1,Tt-1)],Tt,Tt)*Kh;
    Sy = Kpi'*sparse(id1,id1,expinvh,Tt,Tt)*Kpi;
    Hpi =  Sy + Staupi;
    taut = Hpi\(Staupi*alppi + Sy*(yt-mupi));
    taut = min(taut,bpi-.05);    taut = max(taut,api+.05);
    cholHpi = chol(Hpi);
        % AR step
            % compute the constant c
    taustar = taut;
    logc = -sum(log(normcdf((bpi-taustar(1:end-1))/sqrtsigtaupi)-normcdf((api-taustar(1:end-1))/sqrtsigtaupi))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        tauc = taut + cholHpi\randn(Tt,1);
        if  max(tauc) <= bpi - .02 && min(tauc) >= api + .02
            alpARc = -sum(log(normcdf((bpi-tauc(1:end-1))/sqrtsigtaupi)-normcdf((api-tauc(1:end-1))/sqrtsigtaupi))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end
    if flag == 1
        alpAR = - sum(log(normcdf((bpi-taupi(1:end-1))/sqrtsigtaupi)-normcdf((api-taupi(1:end-1))/sqrtsigtaupi))) - logc;
        if isim == 1
            alpMH=1;
        elseif alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end
        if alpMH > log(rand)
            taupi = tauc;
        end
    end
    
        % sample tauu
    Lam = sparse(id1,id1,-lam);
    pistar = yt-taupi;
    z = pistar - rhopi.*[y0;pistar(id3)] - lam.*ut;
    tempSpi = Lam'*sparse(id1,id1,expinvh);
    Spi = tempSpi*Lam;
    Ku = spdiags([ones(Tt,1) [-rhou(1)*ones(Tt-1,1); 0] [-rhou(2)*ones(Tt-2,1);0;0]], [0 -1 -2], Tt, Tt);
    muu = Ku\[rhou(1)*(u0(2)-tauu0(2))+rhou(2)*(u0(1)-tauu0(1)); rhou(2)*(u0(2)-tauu0(2)); sparse(Tt-2,1)];
    alpu = Kh\[tauu0(2); sparse(Tt-1,1)];
    Su  = Ku'*sparse(id1,id1,[1/5 1/5 repmat(1/sigu,1,Tt-2)])*Ku;     %% u1 and u2 have variances 5
    Stauu = Kh'*sparse(id1,id1,[invVtauu repmat(1/sigtauu,1,Tt-1)])*Kh;
    Hu =  Su + Stauu + Spi;
    taut = Hu\(Stauu*alpu + Su*(ut-muu) + tempSpi*z);
    taut = min(taut,bu-.05);    taut = max(taut,au+.05);
    cholHu = chol(Hu);
        % AR step
            % compute the constant c
    taustar = taut;
    logc = -sum(log(normcdf((bu-taustar(1:end-1))/sqrtsigtauu)-normcdf((au-taustar(1:end-1))/sqrtsigtauu))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        tauc = taut + cholHu\randn(Tt,1);
        if  max(tauc) <= bu - .02 && min(tauc) >= au + .02
            alpARc =  -sum(log(normcdf((bu-tauc(1:end-1))/sqrtsigtauu)-normcdf((au-tauc(1:end-1))/sqrtsigtauu))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end
    if flag == 1
        alpAR = - sum(log(normcdf((bu-tauu(1:end-1))/sqrtsigtauu)-normcdf((au-tauu(1:end-1))/sqrtsigtauu))) - logc;
        if isim == 1
            alpMH = 1;
        elseif alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end
        if alpMH > log(rand)
            tauu = tauc;
        end
    end
    
        % sample rhopi
    pistar = yt - taupi;
    ustar = ut - tauu;
    XinvSy = [y0-taupi0; pistar(id3)].*expinvh;
    XinvSyX = XinvSy.*[(y0-taupi0); pistar(id3)];
    Hrhopi = sparse(id1,id1,XinvSyX) + Kh'*sparse(id1,id1,[invVrhopi repmat(1/sigrhopi,1,Tt-1)])*Kh;
    rhot = Hrhopi\(XinvSy.*(pistar+Lam*ustar));
    rhot = min(rhot,1-.05);    rhot = max(rhot,.02);
    cholHrhopi = chol(Hrhopi);
        % AR step
            % compute the constant c
    sqrtsigrhopi = sqrt(sigrhopi);
    rhostar = rhot;
    logc = -sum(log(normcdf((1-rhostar(1:end-1))/sqrtsigrhopi)-normcdf(-rhostar(1:end-1)/sqrtsigrhopi))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        rhoc = rhot + cholHrhopi\randn(Tt,1);
        if  max(rhoc)<= 1 -.005 && min(rhoc)>= .01
            alpARc = -sum(log(normcdf((1-rhoc(1:end-1))/sqrtsigrhopi)-normcdf(-rhoc(1:end-1)/sqrtsigrhopi))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end
    if flag == 1
        alpAR = -sum(log(normcdf((1-rhopi(1:end-1))/sqrtsigrhopi)-normcdf(-rhopi(1:end-1)/sqrtsigrhopi))) - logc;
        if isim == 1
            alpMH = 1;
        elseif alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end
        if alpMH > log(rand)
            rhopi = rhoc;
        end
    end
    
        % sample rhou
    Xu = [[u0(2)-tauu0(2); ustar(id3)] [u0-tauu0; ustar(1:Tt-2)]];
    XuinvSu = Xu'*sparse(1:Tt,1:Tt,[1/5 1/5 1/sigu*ones(1,(Tt-2))]);
    invDrhou = invVrhou + XuinvSu*Xu;
    drhou = XuinvSu*ustar;
    rhouhat = invDrhou\drhou;
    cholHrhou = chol(invDrhou);
    flag = 0;
    while flag==0
        rhou = rhouhat + cholHrhou\randn(2,1);
        if sum(rhou)<1 && rhou(2)-rhou(1)<1 && abs(rhou(2))<1
            flag = 1;
        end
    end
    
        % sample lam
    w = pistar - rhopi.*[y0-taupi0; pistar(id3)];
    XinvSy = ustar.*expinvh;
    XinvSyX = XinvSy.*ustar;
    Hlam = sparse(id1,id1,XinvSyX) + Kh'*sparse(id1,id1,[invVlam repmat(1/siglam,1,Tt-1)])*Kh;
    lamt = Hlam\(XinvSy.*w);
    lamt = max(lamt,-1+.05);    lamt = min(lamt,-.05);
    cholHlam = chol(Hlam);
        % AR step
            % compute the constant c
    sqrtsiglam = sqrt(siglam);
    lamstar = lamt;
    logc = -sum(log(normcdf(-lamstar(1:end-1)/sqrtsiglam)-normcdf((-1-lamstar(1:end-1))/sqrtsiglam))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        lamc = lamt + cholHlam\randn(Tt,1);
        if  max(lamc)<= -.01 && min(lamc)>= -1+.01
            alpARc = -sum(log(normcdf((-lamc(1:end-1))/sqrtsiglam)-normcdf((-1-lamc(1:end-1))/sqrtsiglam))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end
    if flag == 1
        alpAR = -sum(log(normcdf(-lam(1:end-1)/sqrtsiglam)-normcdf((-1-lam(1:end-1))/sqrtsiglam))) - logc;
        if alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end
        if alpMH > log(rand)
            lam = lamc;
        end
    end
    
        % sample h
    Sh = spdiags([invVh; 1/sigh*ones(Tt-1,1)],0,Tt,Tt);
    KSKh = Kh'*Sh*Kh;
    s2 = (w-lam.*ustar).^2;
    errh = 1; ht = h;
    while errh> 10^(-3)
        expht = exp(ht);
        sexpht = s2./expht;
        fh = -.5 + .5*sexpht;
        Gh = .5*sexpht;
        Hh = KSKh + spdiags(Gh,0,Tt,Tt);
        newht = Hh\(fh+Gh.*ht);
        errh = max(abs(newht-ht));
        ht = newht;
    end
    cholHh = chol(Hh);
        % AR-step:
    hstar = ht;
    logc = -.5*hstar'*KSKh*hstar -.5*sum(hstar) - .5*exp(-hstar)'*s2 + log(3);
    flag = 0;
    while flag == 0
        hc = ht + cholHh\randn(Tt,1);
        vhc = hc-ht;
        alpARc = -.5*hc'*KSKh*hc -.5*sum(hc) -.5*exp(-hc)'*s2 + ...
            + .5*vhc'*Hh*vhc - logc;
        if alpARc > log(rand)
            flag = 1;
        end
    end
            % MH-step
    vh = h-ht;
    alpAR = -.5*h'*KSKh*h -.5*sum(h) -.5*exp(-h)'*s2 +...
        .5*vh'*Hh*vh - logc;
    if isim == 1
        alpMH =1;
    elseif alpAR < 0
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end
    if alpMH > log(rand)
        h = hc;
    end
    
        % sample sigtaupi
    errsigtaupi = taupi(2:end) - taupi(1:end-1);
    newStaupi = Staupi0 + sum(errsigtaupi.^2);
    sigtaupic = 1/gamrnd(newnutaupi/2, 2./newStaupi);
    sqrtsigtaupic = sqrt(sigtaupic);
    sqrtsigtaupi = sqrt(sigtaupi);
    sigapi = api - taupi(1:end-1);
    sigbpi = bpi - taupi(1:end-1);
    
    alpMH = -sum(log(normcdf(sigbpi/sqrtsigtaupic)-normcdf(sigapi/sqrtsigtaupic))) + ...
        sum(log(normcdf(sigbpi/sqrtsigtaupi)-normcdf(sigapi/sqrtsigtaupi)));
    if alpMH > log(rand)
        sigtaupi = sigtaupic;
        sqrtsigtaupi = sqrtsigtaupic;
    end
    
        % sample sigtauu
    errsigtauu = tauu(2:end) - tauu(1:end-1);
    newStauu = Stauu0 + sum(errsigtauu.^2);
    sigtauuc = 1/gamrnd(newnutauu/2, 2./newStauu);
    sqrtsigtauuc = sqrt(sigtauuc);
    sqrtsigtauu = sqrt(sigtauu);
    sigau = au - tauu(1:end-1);
    sigbu = bu - tauu(1:end-1);
    
    alpMH = -sum(log(normcdf(sigbu/sqrtsigtauuc)-normcdf(sigau/sqrtsigtauuc))) + ...
        sum(log(normcdf(sigbu/sqrtsigtauu)-normcdf(sigau/sqrtsigtauu)));
    if alpMH > log(rand)
        sigtauu = sigtauuc;
        sqrtsigtauu = sqrtsigtauuc;
    end
    
        % sample sigrhopi
    if tvp_rhopi == 1
        errsigrhopi = rhopi(2:end) - rhopi(1:end-1);
        newSrhopi = Srhopi0 + sum(errsigrhopi.^2);
        sigrhopic = 1/gamrnd(newnurhopi/2, 2./newSrhopi);
        sqrtsigrhopic = sqrt(sigrhopic);
        sqrtsigrhopi = sqrt(sigrhopi);
        sigapi = -rhopi(1:end-1);
        sigbpi = 1-rhopi(1:end-1);
        alpMH = -sum(log(normcdf(sigbpi/sqrtsigrhopic)-normcdf(sigapi/sqrtsigrhopic))) + ...
            sum(log(normcdf(sigbpi/sqrtsigrhopi)-normcdf(sigapi/sqrtsigrhopi)));
        if alpMH > log(rand)
            sigrhopi = sigrhopic;
            sqrtsigrhopi = sqrtsigrhopic;
        end
    end
    
        % sample siglam
    if tvp_lam == 1    
        errsiglam = lam(2:end) - lam(1:end-1);
        newSlam = Slam0 + sum(errsiglam.^2);
        siglamc = 1/gamrnd(newnulam/2, 2./newSlam);
        sqrtsiglamc = sqrt(siglamc);
        sqrtsiglam = sqrt(siglam);
        sigau = -1-lam(1:end-1);
        sigbu = -lam(1:end-1);
        alpMH = -sum(log(normcdf(sigbu/sqrtsiglamc)-normcdf(sigau/sqrtsiglamc))) + ...
            sum(log(normcdf(sigbu/sqrtsiglam)-normcdf(sigau/sqrtsiglam)));
        if alpMH > log(rand)
            siglam = siglamc;
            sqrtsiglam = sqrtsiglamc;
        end
    end    
    
        % sample sigh
    errh = h(2:end) - h(1:end-1);
    newS = Sh0 + sum(errh.^2);
    sigh = 1/gamrnd(newnuh/2, 2./newS);
    
        % sample sigu
    erru = ustar - Xu*rhou;
    newS = Su0 + sum(erru(3:end).^2);
    sigu = 1/gamrnd(newnuu/2, 2./newS);
    
    if isim>burnin 
            % compute various forecasts
        i = isim-burnin;
        taupitp1 = taupi(end); tauutp1 = tauu(end); tauut = tauu(end-1); htp1 = h(end);
        rhopitp1 = rhopi(end); lamtp1 = lam(end);
        ytp1 = yt(end); utp1 = ut(end); utt = ut(end-1);
        sqsigtaupi = sqrt(sigtaupi); sqsigtau = sqrt(sigtauu); sqsigh = sqrt(sigh);
        sqsigrhopi = sqrt(sigrhopi); sqsiglam = sqrt(siglam); sqsigu = sqrt(sigu);
        for tt=1:16
            errut = rhou(1)*(utp1-tauutp1) + rhou(2)*(utt-tauut);
            errpit = ytp1- taupitp1;
            taupitp1 = tnormrnd(taupitp1,sigtaupi,api,bpi,1);
            tauut = tauutp1;
            tauutp1 = tnormrnd(tauutp1,sigtauu,au,bu,1);
            htp1 = htp1 + sqsigh*randn;
            rhopitp1 = tnormrnd(rhopitp1,sigrhopi,0,1,1);
            lamtp1 = tnormrnd(lamtp1,siglam,-1,0,1);
            utt = utp1;
            um = tauutp1 + errut;
            utp1 = um + sqsigu*randn;
            ym = taupitp1 + rhopitp1*errpit + lamtp1*(utp1-tauutp1);
            ytp1 = ym + exp(htp1/2)*randn;
            if tt == 1
                tmpyhat1(i,:) = [ym um normpdf(y(t+1),ym,exp(htp1/2)) normpdf(u(t+1),um,sqsigu) normpdf(y(t+1),ym,exp(htp1/2))*normpdf(u(t+1),um,sqsigu)];
            elseif tt == 4 && t<=T-tt
                tmpyhat4(i,:) = [ym um normpdf(y(t+4),ym,exp(htp1/2)) normpdf(u(t+4),um,sqsigu) normpdf(y(t+4),ym,exp(htp1/2))*normpdf(u(t+4),um,sqsigu)];
            elseif tt == 8 && t<=T-tt
                tmpyhat8(i,:) = [ym um normpdf(y(t+8),ym,exp(htp1/2)) normpdf(u(t+8),um,sqsigu) normpdf(y(t+8),ym,exp(htp1/2))*normpdf(u(t+8),um,sqsigu)];
            elseif tt == 12 && t<=T-tt
                tmpyhat12(i,:) = [ym um normpdf(y(t+12),ym,exp(htp1/2)) normpdf(u(t+12),um,sqsigu) normpdf(y(t+12),ym,exp(htp1/2))*normpdf(u(t+12),um,sqsigu)];
            elseif tt == 16 && t<=T-tt
                tmpyhat16(i,:) = [ym um normpdf(y(t+16),ym,exp(htp1/2)) normpdf(u(t+16),um,sqsigu) normpdf(y(t+16),ym,exp(htp1/2))*normpdf(u(t+16),um,sqsigu)];
            end
        end
    end
end
 