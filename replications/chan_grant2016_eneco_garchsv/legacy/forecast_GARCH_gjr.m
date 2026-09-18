% This script computes the forecasts from the GARCH-GJR model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH-GJR';
warning off;
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute and define a few things
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
mu = mean(yt);
e = yt-mu;
e2 = e.^2;
Z = [ones(Tt-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
sig20 = var(yt);
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
del = 0;
gam = gamt;
while sum(exp(gam(2:3))) > .999 
    gam(2:end) = log(.99*exp(gam(2:end)));
end
tmpt = [gam;del];
ybar = mean(yt);
s2 = var(yt)/Tt;

for isim = 1:nsims + burnin  
        % sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec, sig2c] = loglike_garch_gjr(yt-muc,gam,del,sig20);
    [llike,sig2] = loglike_garch_gjr(yt-mu,gam,del,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;        
    end    
        % sample gam and del  
    e = yt-mu;
    [llike,sig2] = loglike_garch_gjr(e,gam,del,sig20);
    if isim == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        tmpt = fminsearch(@(x)-loglike_garch_gjr(e,x(1:3),x(4),sig20)...
            -lpri_gam(x(1:3)),tmpt);        
        gamt = tmpt(1:3);
        delt = tmpt(4);
        expgamt = exp(gamt);
        sig2 = zeros(Tt,1);
        dsig2 = zeros(Tt,4);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;
        for t=2:Tt
            sig2(t) = expgamt(1) + (expgamt(2)+delt*(e(t-1)<0))*e2(t-1) ...
                + expgamt(3)*sig2(t-1);
            dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1);
            dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2);
            dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3));
            dsig2(t,4) = (e(t-1)<0)*e2(t-1) + expgamt(3)*dsig2(t-1,4);
        end            
        S = repmat(.5./sig2.*(e2./sig2-1),1,4) .* dsig2;
        I = S'*S;        
        lprop = @(x) -.5*(x-tmpt)'*I*(x-tmpt);
        Ctmp = chol(I,'lower');
    end    
    tmpc = tmpt + Ctmp'\randn(4,1);
    gamc = tmpc(1:3);    
    delc = tmpc(4);    
    if sum(exp(gamc(2:3))) < .999 %% impose stationarity
        [llikec,sig2c] = loglike_garch_gjr(e,gamc,delc,sig20);
        alpMH = llikec + lpri_gam(gamc) + log(1/(1-gamc(3))) - lprop([gamc;delc])...
            - (llike + lpri_gam(gam) + log(1/(1-gam(3))) - lprop([gam;del]));
        if alpMH > log(rand)
            del = delc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;            
        end
    end
    
    if isim > burnin
        isave = isim - burnin;        
        et = yt(end)-mu;
        sig2tp1 = exp(gam(1)) + (exp(gam(2))+del*(et<0))*et^2 + exp(gam(3))*sig2(end);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-mu).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

