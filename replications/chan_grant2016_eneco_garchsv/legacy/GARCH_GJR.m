% This script estimates the GARCH-GJR model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
mu0 = 0; Vmu = 10;
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
        % estimate stationarity prob
tempgam = repmat(gam0',10000,1) + (chol(Vgam,'lower')*randn(3,10000))';
prob_s = sum(sum(exp(tempgam(:,2:end)),2)<1)/10000;
c_prior = -.5*log(2*pi*Vmu) -3/2*log(2*pi) -.5*log(det(Vgam)) -log(prob_s);
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
prior = @(m,g,d) c_prior + lpri_mu(m) + lpri_gam(g) + log(1/(1-g(3)));

    %% initialize for storeage
store_theta = zeros(nloop - burnin,5); % mu, gam, del
store_sig2 = zeros(nloop - burnin,T); 
store_llike = zeros(nloop-burnin,1);
store_Q = zeros(nloop - burnin,2);
    %% initialize the Markov chain
mu = mean(y);
e = y-mu;
e2 = e.^2;
Z = [ones(T-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
sig20 = var(y);
gam = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
del = 0;
while sum(exp(gam(2:3))) > .999 
    gam(2:3) = log(.99*exp(gam(2:3)));
end
tmpt = [gam;del];
ybar = mean(y);
s2 = var(y)/T;
countgam = 0;
countmu = 0;

disp('Starting GARCH-GJR.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
        %% sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec sig2c] = loglike_garch_gjr(y-muc,gam,del,sig20);
    [llike sig2] = loglike_garch_gjr(y-mu,gam,del,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
    
        %% sample gam and del  
    e = y-mu;
    [llike sig2] = loglike_garch_gjr(e,gam,del,sig20);
    if loop < 10 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        tmpt = fminsearch(@(x)-loglike_garch_gjr(e,x(1:3),x(4),sig20)...
            -lpri_gam(x(1:3)),tmpt);        
        gamt = tmpt(1:3);
        delt = tmpt(4);
        expgamt = exp(gamt);
        sig2 = zeros(T,1);
        dsig2 = zeros(T,4);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;
        for t=2:T
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
    if sum(exp(gamc(2:3))) < .999 || loop < 5 %% impose stationarity
        [llikec sig2c] = loglike_garch_gjr(e,gamc,delc,sig20);
        alpMH = llikec + lpri_gam(gamc) + log(1/(1-gamc(3))) - lprop([gamc;delc])...
            - (llike + lpri_gam(gam) + log(1/(1-gam(3))) - lprop([gam;del]));
        if alpMH > log(rand) || loop < 5
            del = delc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;
            countgam = countgam + 1;        
        end
    end
    if loop>burnin
        i = loop-burnin;        
        store_theta(i,:) = [mu exp(gam)' del];
        store_sig2(i,:) = sig2';
        store_llike(i) = llike; 
        
            % compute Q stats 
        u = (y-mu)./sqrt(sig2);
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
    [ml mlstd] = ml_garch_gjr(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('delta_1     | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end

