% This script computes the forecasts from the GARCH-MA model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH-MA';
warning off;
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]

    % compute and define a few things
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_psi = @(x) -.5*(x-psi0)^2/Vpsi;
sig20 = var(yt);
mu = mean(yt);
lpsi = @(x) loglike_garch_ma(yt-mu,x,log([var(yt) .1 .5])',sig20) + lpri_psi(x);
psi = fminsearch(@(x)-lpsi(x),0);
psihat = psi;
Hpsi = speye(Tt) + sparse(2:Tt,1:(Tt-1),psi*ones(1,Tt-1),Tt,Tt); 
e = Hpsi\yt;
e2 = e.^2;
Z = [ones(Tt-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
gam = gamt;
while sum(exp(gam(2:3))) > .999 
    gam(2:end) = log(.99*exp(gam(2:end)));
end
ybar = mean(yt);
s2 = var(yt)/Tt;

for isim = 1:nsims + burnin  
        % sample mu
    Xmu = Hpsi\ones(Tt,1);
    muhat = (Xmu'*Xmu)\(Xmu'*(Hpsi\yt));
    s2 = sum((Hpsi\(yt-muhat)).^2)/Tt/(Xmu'*Xmu);    
    muc = muhat + sqrt(s2)*randn;    
    [llikec,sig2c] = loglike_garch_ma(yt-muc,psi,gam,sig20);
    [llike,sig2] = loglike_garch_ma(yt-mu,psi,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-muhat)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-muhat)^2/s2);
    if alpMH > log(rand)
        mu = muc;        
    end
        % sample psi    
    lpsi = @(x) loglike_garch_ma(yt-mu,x,gam,sig20) + lpri_psi(x);    
    psihat = fminsearch(@(x)-lpsi(x),psihat);   
    sqVpsic = .05; Vpsic = sqVpsic^2;
    psic = psihat + sqVpsic*randn;
    if abs(psic)<.999
        alpMH = lpsi(psic) - lpsi(psi) ...
            -.5*(psi-psihat)^2/Vpsic + .5*(psic-psihat)^2/Vpsic;
    else
        alpMH = -inf;
    end
    if alpMH>log(rand)
        psi = psic;
        Hpsi = speye(Tt) + sparse(2:Tt,1:(Tt-1),psi*ones(1,Tt-1),Tt,Tt);         
    end   
        % sample gam    
    [llike,sig2] = loglike_garch_ma(yt-mu,psi,gam,sig20);
    if isim == 1 || rand>.5  %% maximize with prob 0.5
        e = Hpsi\yt;
        e2 = e.^2;                
        gamt = fminsearch(@(x)-loglike_garch_ma(yt-mu,psi,x,sig20)...
            -lpri_gam(x),gamt);
        expgamt = exp(gamt);
        sig2 = zeros(Tt,1);
        dsig2 = zeros(Tt,3);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;
        for t=2:Tt
            sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) + expgamt(3)*sig2(t-1);
            dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1);
            dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2);
            dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3));
        end
        S = repmat(.5./sig2.*(e2./sig2-1),1,3) .* dsig2;
        I = S'*S;        
        lprop = @(x) -.5*(x-gamt)'*I*(x-gamt);
        Cgam = chol(I,'lower');
    end
    count = 0;
    gamc = gamt + Cgam'\randn(3,1);
    while sum(exp(gamc(2:end))) > .999 && count < 100
        gamc = gamt + Cgam'\randn(3,1);
        count = count + 1;
    end
    [llikec,sig2c] = loglike_garch_ma(yt-mu,psi,gamc,sig20); 
    alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
        - (llike + lpri_gam(gam) - lprop(gam));
    if alpMH > log(rand)
        gam = gamc;
        sig2 = sig2c;        
    end
    
    if isim > burnin
        isave = isim - burnin; 
        u = Hpsi\(yt-mu);
        et = yt(end)-mu;
        ym = mu + psi*u(end);
        sig2tp1 = exp(gam(1)) + exp(gam(2))*et^2 + exp(gam(3))*sig2(end);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-ym).^2/sig2tp1; 
        tempyhat1(isave,:) = [ym lden]; 
    end
end

