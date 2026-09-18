% This script estimates the GARCH-J model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.


    %% prior
mu0 = 0; Vmu = 10;
delta0 = [0 log(10)]'; Vdelta = diag([10 1]);
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
        % estimate stationarity prob
tempgam = repmat(gam0',10000,1) + (chol(Vgam,'lower')*randn(3,10000))';
prob_s = sum(sum(exp(tempgam(:,2:end)),2)<1)/10000;
c_prior = -.5*log(2*pi*Vmu) - log(2*pi) -.5*log(det(Vdelta)) ...
    -3/2*log(2*pi) -.5*log(det(Vgam)) -log(prob_s);
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_kappa = @(x) log(1/(.1-0)) - 10^(10)*(x>.1);
lpri_delta = @(x) -.5*(x-delta0)'*(Vdelta\(x-delta0));
prior = @(m,d,k,g) c_prior + lpri_mu(m) + lpri_delta(d) ...
    + lpri_kappa(k) + lpri_gam(g);

    %% initialize for storeage
store_theta = zeros(nloop - burnin,7); % mu, delta, kappa, gam
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
gamt = fminsearch(@(x)-loglike_garch(e,x,sig20)-lpri_gam(x),[tempb; tempb(2)]);
expgamt = exp(gamt);
sig2 = zeros(T,1);
dsig2 = zeros(T,3);
dsig2(1,1) = expgamt(1);
dsig2(1,3) = expgamt(3)*sig20;
sig2(1) = expgamt(1) + expgamt(3)*sig20;
for t=2:T
    sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) + expgamt(3)*sig2(t-1);
    dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1);
    dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2);
    dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3));
end
S = repmat(.5./sig2.*(e2./sig2-1),1,3) .* dsig2;
I = S'*S;    
lprop = @(x) -.5*(x-gamt)'*I*(x-gamt);
Cgam = chol(I,'lower');
gam = gamt + Cgam'\randn(3,1);    
while sum(exp(gam(2:3))) > .999 
    gam = gamt + Cgam'\randn(3,1);    
end
kappa = .05;
delta = fminsearch(@(x)-loglike_garch_j(y-mu,x,kappa,gam,sig20)...
        -lpri_delta(x),[0 0]');
ybar = mean(y);
s2 = var(y)/T;
del_std = diag([sqrt(s2) 1]);
k_pristd = sqrt((.1-0)^2/12);
countgam = 0;
countmu = 0;
countdelta = 0;
countkappa = 0;

disp('Starting GARCH-J.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
    
        %% sample delta
    deltat = fminsearch(@(x)-loglike_garch_j(y-mu,x,kappa,gam,sig20)...
         -lpri_delta(x),delta);    
    deltac = deltat + del_std*randn(2,1); 
    [llikec sig2c] = loglike_garch_j(y-mu,deltac,kappa,gam,sig20);
    [llike sig2] = loglike_garch_j(y-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_delta(deltac) + .5*sum((del_std\(deltac-deltat)).^2) ...
        - (llike + lpri_delta(delta) + .5*sum((del_std\(delta-deltat)).^2));
    if alpMH > log(rand)
        delta = deltac;
        countdelta = countdelta + 1;        
    end
    
        %% sample kappa
    kappat = fminbnd(@(x)-loglike_garch_j(y-mu,delta,x,gam,sig20)...
        -lpri_kappa(x),.001,.0999);
    kappac = kappat + k_pristd*randn;
    while kappac > .1 || kappac < .001
        kappac = kappat + k_pristd*randn;
    end
    [llikec sig2c] = loglike_garch_j(y-mu,delta,kappac,gam,sig20);
    [llike sig2] = loglike_garch_j(y-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_kappa(kappac) + .5*(kappac-kappat)^2/k_pristd^2 ...
        - (llike + lpri_kappa(kappa) + .5*(kappa-kappat)^2/k_pristd^2);
    if alpMH > log(rand)
        kappa = kappac;
        countkappa = countkappa + 1;        
    end   
    
        %% sample mu
    ybar = fminsearch(@(x)-loglike_garch_j(y-x,delta,kappa,gam,sig20)...
        -lpri_mu(x),mu);
    muc = ybar + sqrt(s2)*randn;
    [llikec sig2c] = loglike_garch_j(y-muc,delta,kappa,gam,sig20);
    [llike sig2] = loglike_garch_j(y-mu,delta,kappa,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;         
    end
    
        %% sample gam      
    e = y-mu;
    [llike sig2] = loglike_garch_j(e,delta,kappa,gam,sig20);    
    if loop == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        gamt = fminsearch(@(x)-loglike_garch_j(e,delta,kappa,x,sig20)-lpri_gam(x),gamt);
        
        expgamt = exp(gamt);
        sig2 = zeros(T,1);
        dsig2 = zeros(T,3);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;
        for t=2:T
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
        [llikec sig2c] = loglike_garch_j(e,delta,kappa,gamc,sig20);
        alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
            - (llike + lpri_gam(gam) - lprop(gam));
    else 
        alpMH = -10^100;
    end
    if alpMH > log(rand) || loop < 10
        gam = gamc;
        sig2 = sig2c;
        llike = llikec;
        countgam = countgam + 1;        
    end    
    if loop>burnin
        i = loop-burnin;        
        store_theta(i,:) = [mu delta(1) exp(delta(2)) kappa exp(gam)'];
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
accept = [countmu/nloop countdelta/nloop countkappa/nloop countgam/nloop];
Qhat = mean(store_Q)';
Qstd = std(store_Q)';

figure;    
plot(tid, sighat, 'LineWidth',2,'Color','black'); box off;
title('\sigma_t');

%% compute ml
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');    
    [ml mlstd] = ml_garch_j(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(6), thetastd(6)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(7), thetastd(7)); 
fprintf('kappa       | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('mu_k        | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('sigma_k^2   | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 


if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end