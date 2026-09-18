% This script computes the forecasts from the standard GARCH(1,1) model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH';
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]
    % compute and define a few things
warning off;
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
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
ybar = mean(yt);
s2 = var(yt)/Tt;

for isim = 1:nsims + burnin  
        % sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec,sig2c] = loglike_garch(yt-muc,gam,sig20);
    [llike,sig2] = loglike_garch(yt-mu,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;        
    end    
        % sample gam      
    e = yt-mu;
    [llike,sig2] = loglike_garch(e,gam,sig20);    
    if isim == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),gamt);
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
        newgamt = gamt + I\sum(S)';    
        lprop = @(x) -.5*(x-gamt)'*I*(x-gamt);
        Cgam = chol(I,'lower');
    end    
    gamc = gamt + Cgam'\randn(3,1);    
    if sum(exp(gamc(2:3))) < .999 %% impose stationarity
        [llikec,sig2c] = loglike_garch(e,gamc,sig20);
        alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
            - (llike + lpri_gam(gam) - lprop(gam));
        if alpMH > log(rand)
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;            
        end
    end
    
    if isim > burnin
        isave = isim - burnin;        
        et = yt(end)-mu;
        sig2tp1 = exp(gam(1)) + exp(gam(2))*et^2 + exp(gam(3))*sig2(end);
        lden = -.5*log(2*pi*sig2tp1) - .5*(y(t+1)-mu).^2/sig2tp1; 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

