% This script computes the forecasts from the GARCH-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH-J';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute and define a few things
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_kappa = @(x) log(1/(.1-0)) - 10^(10)*(x>.1);
lpri_delta = @(x) -.5*(x-delta0)'*(Vdelta\(x-delta0));
mu = mean(yt);
e = yt-mu;
e2 = e.^2;
Z = [ones(Tt-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
sig20 = var(yt);
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
gam = gamt;
while sum(exp(gam(2:3))) > .999 
    gam(2:end) = log(.99*exp(gam(2:end)));
end
kappa = .05;
delta = fminsearch(@(x)-loglike_garch_j(yt-mu,x,kappa,gam,sig20)...
        -lpri_delta(x),[0 0]');
ybar = mean(yt);
s2 = var(yt)/Tt;
del_std = diag([sqrt(s2) 1]);
k_pristd = sqrt((.1-0)^2/12);

for isim = 1:nsims + burnin  
        % sample delta
    deltat = fminsearch(@(x)-loglike_garch_j(yt-mu,x,kappa,gam,sig20)...
         -lpri_delta(x),delta);    
    deltac = deltat + del_std*randn(2,1); 
    [llikec,sig2c] = loglike_garch_j(yt-mu,deltac,kappa,gam,sig20);
    [llike,sig2] = loglike_garch_j(yt-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_delta(deltac) + .5*sum((del_std\(deltac-deltat)).^2) ...
        - (llike + lpri_delta(delta) + .5*sum((del_std\(delta-deltat)).^2));
    if alpMH > log(rand)
        delta = deltac;    
    end    
        % sample kappa
    kappat = fminbnd(@(x)-loglike_garch_j(yt-mu,delta,x,gam,sig20)...
        -lpri_kappa(x),.001,.0999);
    kappac = kappat + k_pristd*randn;
    while kappac > .1 || kappac < .001
        kappac = kappat + k_pristd*randn;
    end
    [llikec,sig2c] = loglike_garch_j(yt-mu,delta,kappac,gam,sig20);
    [llike,sig2] = loglike_garch_j(yt-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_kappa(kappac) + .5*(kappac-kappat)^2/k_pristd^2 ...
        - (llike + lpri_kappa(kappa) + .5*(kappa-kappat)^2/k_pristd^2);
    if alpMH > log(rand)
        kappa = kappac;        
    end    
        % sample mu
    ybar = fminsearch(@(x)-loglike_garch_j(yt-x,delta,kappa,gam,sig20)...
        -lpri_mu(x),mu);
    muc = ybar + sqrt(s2)*randn;
    [llikec,sig2c] = loglike_garch_j(yt-muc,delta,kappa,gam,sig20);
    [llike,sig2] = loglike_garch_j(yt-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;        
    end    
        % sample gam      
    e = yt-mu;
    [llike,sig2] = loglike_garch_j(e,delta,kappa,gam,sig20);    
    if isim == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        gamt = fminsearch(@(x)-loglike_garch_j(e,delta,kappa,x,sig20)-lpri_gam(x),gamt);        
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
    if count<100
        [llikec,sig2c] = loglike_garch_j(e,delta,kappa,gamc,sig20);
        alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
            - (llike + lpri_gam(gam) - lprop(gam));
    else 
        alpMH = -10^100;
    end
    if alpMH > log(rand) || isim < 10
        gam = gamc;
        sig2 = sig2c;
        llike = llikec;        
    end    
    
    if isim > burnin
        isave = isim - burnin;
        et = yt(end)-mu;
        sig2tp1 = exp(gam(1)) + exp(gam(2))*et^2 + exp(gam(3))*sig2(end);
        ktp1 = delta(1) + exp(delta(2)/2)*randn;
        ym = mu + (kappa>rand)*ktp1;
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-ym).^2/sig2tp1; 
        tempyhat1(isave,:) = [ym lden]; 
    end
end

