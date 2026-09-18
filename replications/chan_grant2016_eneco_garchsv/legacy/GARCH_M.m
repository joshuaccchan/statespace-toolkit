% This script estimates the GARCH-M model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
mu0 = 0; Vmu = 10;
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
lam0 = 0; Vlam = 100;
lpri_lam = @(x) -.5*(x-lam0)^2/Vlam;
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
c_prior = -.5*log(2*pi*Vmu) -.5*log(2*pi*Vlam) -3/2*log(2*pi) -.5*log(det(Vgam));
prior = @(m,l,g) c_prior + lpri_mu(m) + lpri_lam(l) + lpri_gam(g);

    %% initialize for storeage
store_theta = zeros(nloop - burnin,5); % mu, lam, gam
store_sig2 = zeros(nloop - burnin,T); 
store_llike = zeros(nloop-burnin,1);
store_Q = zeros(nloop - burnin,2);
    %% initialize the Markov chain
sig20 = var(y);
mu = mean(y);
e = y-mu;
e2 = e.^2;
Z = [ones(T-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
gam = gamt;
thetat = fminsearch(@(x) -loglike_garch_m(y-x(1),x(3:5),x(2),sig20)...
    -lpri_gam(x(2:4))-lpri_lam(x(1)),[mean(y);.001;gam]);
tempt = thetat(2:end);
mu = thetat(1);
lam = thetat(2);
gam = thetat(3:end);
[llike sig2] = loglike_garch_m(y-mu,gam,lam,sig20);
s2 = var(y-mu-lam*sig2)/T;
countmu = 0;
countgam = 0;

disp('Starting GARCH-M.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
    
        %% sample mu    
    muhat = fminsearch(@(x) -loglike_garch_m(y-x,gam,lam,sig20),mu);
    muc = muhat + sqrt(s2)*randn;    
    [llike sig2] = loglike_garch_m(y-mu,gam,lam,sig20); 
    [llikec sig2c] = loglike_garch_m(y-muc,gam,lam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-muhat)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-muhat)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
    
        %% sample lam and gam
    e = y-mu;
    [llike sig2] = loglike_garch_m(e,gam,lam,sig20);
    if loop == 1 || rand>.5  %% maximize with prob 0.2
        tempt = fminsearch(@(x)-loglike_garch_m(y-mu,x(2:4),x(1),sig20)...
            -lpri_gam(x(2:4))-lpri_lam(x(1)),tempt);
        lamt = tempt(1);
        gamt = tempt(2:end);         
        expgamt = exp(gamt);
        sig2 = zeros(T,1);
        dsig2 = zeros(T,4);
        dsig2(1,2) = expgamt(1);
        dsig2(1,4) = expgamt(3)*sig20;
        sig2(1) = expgamt(1)+expgamt(3)*sig20;
        for t=2:T
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
        [llikec sig2c] = loglike_garch_m(e,gamc,lamc,sig20);
        alpMH = llikec + lpri_gam(gamc) + lpri_lam(lamc) - lprop([lamc;gamc])...
            - (llike + lpri_gam(gam) + lpri_lam(lam) - lprop([lam;gam]));
        if alpMH > log(rand)
            lam = lamc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;
            countgam = countgam + 1;        
        end
    end 

    if loop>burnin
        i = loop-burnin;        
        store_theta(i,:) = [mu lam exp(gam)'];
        store_sig2(i,:) = sig2';
        store_llike(i) = llike; 
        
            % compute Q stats 
        u = (y-mu-lam*sig2)./sqrt(sig2);
        rtmp = autocorr(u,nlag);
        Q = T*((T+2)./(T-(1:nlag)))*rtmp(2:end).^2;
        rtmp = autocorr(u.^2,nlag);
        Q2 = T*((T+2)./(T-(1:nlag)))*rtmp(2:end).^2;
        store_Q(i,:) = [Q Q2];
    end    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
sighat = mean(sqrt(store_sig2))';  % plot std. dev.
thetastd = std(store_theta)';
accept = [countmu/nloop countgam/nloop];
Qhat = mean(store_Q)';
Qstd = std(store_Q)';

figure;    
plot(tid, sighat, 'LineWidth',2,'Color','black'); box off;
title('\sigma_t');

%% compute ml
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');    
    [ml mlstd] = ml_garch_m(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('lambda      | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end

