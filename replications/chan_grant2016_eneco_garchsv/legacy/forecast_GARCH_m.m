% This script computes the forecasts from the GARCH-M model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH-M';
warning off;
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute and define a few things
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_lam = @(x) -.5*(x-lam0)^2/Vlam;

sig20 = var(yt);
mu = mean(yt);
e = yt-mu;
e2 = e.^2;
Z = [ones(Tt-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
gam = gamt;
while sum(exp(gam(2:3))) > .999 
    gam(2:end) = log(.99*exp(gam(2:end)));
end
thetat = fminsearch(@(x) -loglike_garch_m(yt-x(1),x(3:5),x(2),sig20)...
    -lpri_gam(x(2:4))-lpri_lam(x(1)),[mean(yt);.001;gam]);
tempt = thetat(2:end);
mu = thetat(1);
lam = thetat(2);
gam = thetat(3:end);
[llike,sig2] = loglike_garch_m(yt-mu,gam,lam,sig20);
s2 = var(yt-mu-lam*sig2)/Tt;

for isim = 1:nsims + burnin  
        % sample mu    
    muhat = fminsearch(@(x) -loglike_garch_m(yt-x,gam,lam,sig20),mu);
    muc = muhat + sqrt(s2)*randn;    
    [llike,sig2] = loglike_garch_m(yt-mu,gam,lam,sig20); 
    [llikec,sig2c] = loglike_garch_m(yt-muc,gam,lam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-muhat)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-muhat)^2/s2);
    if alpMH > log(rand)
        mu = muc;        
    end
        % sample lam and gam
    e = yt-mu;
    [llike,sig2] = loglike_garch_m(e,gam,lam,sig20);
    if isim == 1 || rand>.2  %% maximize with prob 0.2
        tempt = fminsearch(@(x)-loglike_garch_m(yt-mu,x(2:4),x(1),sig20)...
            -lpri_gam(x(2:4))-lpri_lam(x(1)),tempt);
        lamt = tempt(1);
        gamt = tempt(2:end);         
        expgamt = exp(gamt);
        sig2 = zeros(Tt,1);
        dsig2 = zeros(Tt,4);
        dsig2(1,2) = expgamt(1);
        dsig2(1,4) = expgamt(3)*sig20;
        sig2(1) = expgamt(1)+expgamt(3)*sig20;
        for t=2:Tt
            et = e(t-1) - lamt*sig2(t-1);
            sig2(t) = expgamt(1) + expgamt(2)*et^2 + expgamt(3)*sig2(t-1);            
            dsig2(t,1) = -2*expgamt(2)*et*(sig2(t-1)+lamt*dsig2(t-1,1))...
                + expgamt(3)*dsig2(t-1,1);            
            dsig2(t,2) = expgamt(1) + expgamt(3)*dsig2(t-1,2);
            dsig2(t,3) = expgamt(2)*et^2 + expgamt(3)*dsig2(t-1,3);
            dsig2(t,4) = expgamt(3)*(sig2(t-1) + dsig2(t-1,4));
        end
        S = repmat(.5./sig2.*(e.^2./sig2-1)+.5*lamt^2,1,4).*dsig2;
        S(:,1) = S(:,1) + .5*lamt^2*dsig2(:,1) + lamt*sig2;            
        I = S'*S;   
        lprop = @(x) -.5*(x-tempt)'*I*(x-tempt);
        Ctemp = chol(I,'lower');
    end
    tempc = tempt + Ctemp'\randn(4,1);
    lamc = tempc(1);
    gamc = tempc(2:end);    
    if sum(exp(gamc(2:3))) < .999 %% impose stationarity
        [llikec,sig2c] = loglike_garch_m(e,gamc,lamc,sig20);
        alpMH = llikec + lpri_gam(gamc) + lpri_lam(lamc) - lprop([lamc;gamc])...
            - (llike + lpri_gam(gam) + lpri_lam(lam) - lprop([lam;gam]));
        if alpMH > log(rand)
            lam = lamc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;            
        end
    end 
    
    if isim > burnin
        isave = isim - burnin;        
        et = yt(end)-mu-lam*sig2(end);
        sig2tp1 = exp(gam(1)) + exp(gam(2))*et^2 + exp(gam(3))*sig2(end);
        ym = mu + lam*sig2tp1;
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-ym).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

