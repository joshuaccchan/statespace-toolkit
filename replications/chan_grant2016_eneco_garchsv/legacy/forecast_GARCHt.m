% This script computes the forecasts from the GARCH-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

model_name = 'GARCH-t';
warning off;
tempyhat1 = zeros(nsims,2);  %% [point forecasts, prelike]

%% compute and define a few things
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_nu = @(x) log(1/(nuub-2)) - 10^(10)*(x<2 || x> nuub);
mu = mean(yt);
e = yt-mu;
e2 = e.^2;
Z = [ones(Tt-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
sig20 = var(yt);
tmpt = fminsearch(@(x)-loglike_garch_t(e,x(1:3),sig20,x(4))...
    -lpri_gam(x(1:3))-lpri_nu(x(4)),[tempb; tempb(2);10]); 
gam = tmpt(1:3);
nu = tmpt(4);
while sum(exp(gam(2:3))) > .999 
    gam(2:end) = log(.99*exp(gam(2:end)));
end
ybar = mean(yt);
s2 = var(yt)/Tt;

for isim = 1:nsims + burnin
  
        %% sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec sig2c] = loglike_garch_t(yt-muc,gam,sig20,nu);
    [llike sig2] = loglike_garch_t(yt-mu,gam,sig20,nu);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
    end
    
        %% sample gam      
    e = yt-mu;
    [llike sig2] = loglike_garch_t(e,gam,sig20,nu);
    if isim == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        tmpt = fminsearch(@(x)-loglike_garch_t(e,x(1:3),sig20,x(4))...
            -lpri_gam(x(1:3))-lpri_nu(x(4)),tmpt);        
        gamt = tmpt(1:3);
        nut = tmpt(4);         
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
        S = [repmat(.5./sig2.*(e2./(sig2*nut)./(1+e2./sig2/nut)-1),1,3) .* dsig2 ...
            -.5*log(1+e2./sig2/nut) + (nut+1)/(2*nut^2)*e2./sig2./(1+e2./sig2/nut)];
        I = S'*S;        
        lprop = @(x) -.5*(x-tmpt)'*I*(x-tmpt);
        Ctmp = chol(I,'lower');
    end    
    tmpc = tmpt + Ctmp'\randn(4,1);
    gamc = tmpc(1:3);    
    nuc = tmpc(4);    
    if sum(exp(gamc(2:3))) < .999 && nuc > 0 %% impose stationarity
        [llikec,sig2c] = loglike_garch_t(e,gamc,sig20,nuc);
        alpMH = llikec + lpri_gam(gamc) + lpri_nu(nuc) - lprop([gamc;nuc])...
            - (llike + lpri_gam(gam) + lpri_nu(nu) - lprop([gam; nu]));
        if alpMH > log(rand)
            nu = nuc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;            
        end
    end
    
    if isim > burnin
        isave = isim - burnin;        
        et = yt(end)-mu;
        sig2tp1 = exp(gam(1)) + exp(gam(2))*et^2 + exp(gam(3))*sig2(end);
        lden = gammaln((nu+1)/2) - gammaln(nu/2) - .5*log(nu*pi*sig2tp1) ...
            -(nu+1)/2*log(1+(y(t+1)-mu).^2/(nu*sig2tp1)); 
        tempyhat1(isave,:) = [mu lden]; 
    end
end

