% This script estimates the GARCH-MA model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
mu0 = 0; Vmu = 10;
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
psi0 = 0; Vpsi = 1;
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_psi = @(x) -.5*(x-psi0)^2/Vpsi;
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
    % estimate stationarity prob
tempgam = repmat(gam0',10000,1) + (chol(Vgam,'lower')*randn(3,10000))';
prob_s = sum(sum(exp(tempgam(:,2:end)),2)<1)/10000;
psi_const = 1/(normcdf(1,psi0,sqrt(Vpsi))-normcdf(-1,psi0,sqrt(Vpsi)));
c_prior = -.5*log(2*pi*Vmu) -.5*log(2*pi*Vpsi) + log(psi_const) ...
    -3/2*log(2*pi) -.5*log(det(Vgam)) -log(prob_s);
prior = @(m,p,g) c_prior + lpri_mu(m) + lpri_psi(p) + lpri_gam(g);

    %% initialize for storeage
store_theta = zeros(nloop - burnin,5); % mu, psi, gam
store_sig2 = zeros(nloop - burnin,T); 
store_Q = zeros(nloop - burnin,2);
    %% compute a few things outside the loop
sig20 = var(y);
mu = mean(y);
lpsi = @(x) loglike_garch_ma(y-mu,x,log([var(y) .1 .5])',sig20) + lpri_psi(x);
psi = fminsearch(@(x)-lpsi(x),0);
psihat = psi;
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
e = Hpsi\y;
e2 = e.^2;
Z = [ones(T-1,1) e2(1:end-1)];
tempb = log((Z'*Z)\(Z'*e2(2:end)));
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
countmu = 0;
countgam = 0;
countpsi = 0;

disp('Starting GARCH-MA.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
        %% sample mu
    Xmu = Hpsi\ones(T,1);
    muhat = (Xmu'*Xmu)\(Xmu'*(Hpsi\y));
    s2 = sum((Hpsi\(y-muhat)).^2)/T/(Xmu'*Xmu);    
    muc = muhat + sqrt(s2)*randn;    
    [llikec sig2c] = loglike_garch_ma(y-muc,psi,gam,sig20);
    [llike sig2] = loglike_garch_ma(y-mu,psi,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-muhat)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-muhat)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
        %% sample psi    
    lpsi = @(x) loglike_garch_ma(y-mu,x,gam,sig20) + lpri_psi(x);    
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
        Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
        countpsi = countpsi + 1;
    end      
        %% sample gam    
    [llike sig2] = loglike_garch_ma(y-mu,psi,gam,sig20);
    if loop == 1 || rand>.5  %% maximize with prob 0.5
        e = Hpsi\y;
        e2 = e.^2;                
        gamt = fminsearch(@(x)-loglike_garch_ma(y-mu,psi,x,sig20)...
            -lpri_gam(x),gamt);        
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
    [llikec sig2c] = loglike_garch_ma(y-mu,psi,gamc,sig20); 
    alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
        - (llike + lpri_gam(gam) - lprop(gam));
    if alpMH > log(rand)
        gam = gamc;
        sig2 = sig2c;
        countgam = countgam + 1;            
    end   
    if loop>burnin
        i = loop-burnin;
        store_theta(i,:) = [mu psi exp(gam)'];        
        store_sig2(i,:) = sig2';
        
            % compute Q stats 
        u = (Hpsi\(y-mu))./sqrt(sig2);
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
accept = [countmu/nloop countpsi/nloop countgam/nloop];
Qhat = mean(store_Q)';
Qstd = std(store_Q)';

figure;    
plot(tid, sighat, 'LineWidth',2,'Color','black'); box off;
title('\sigma_t');

%% compute ml
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');    
    [ml mlstd] = ml_garch_ma(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('psi         | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end


