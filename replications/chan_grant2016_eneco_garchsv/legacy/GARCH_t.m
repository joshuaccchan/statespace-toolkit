% This script estimates the GARCH-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
nuub = 100; %% upperbound for nu
mu0 = 0; Vmu = 10;
gam0 = log([exp(1) .1 .8]'); Vgam = diag([10 1 1]);
        % estimate stationarity prob
tempgam = repmat(gam0',10000,1) + (chol(Vgam,'lower')*randn(3,10000))';
prob_s = sum(sum(exp(tempgam(:,2:end)),2)<1)/10000;
c_prior = -.5*log(2*pi*Vmu) -3/2*log(2*pi) -.5*log(det(Vgam)) -log(prob_s);
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
lpri_nu = @(x) log(1/(nuub-2)) - 10^(10)*(x<2 || x> nuub);
prior = @(m,g,n) c_prior + lpri_mu(m) + lpri_gam(g) + lpri_nu(n);

    %% initialize for storeage
store_theta = zeros(nloop - burnin,5); % mu, gam, nu
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
tmpt = fminsearch(@(x)-loglike_garch_t(e,x(1:3),sig20,x(4))...
    -lpri_gam(x(1:3))-lpri_nu(x(4)),[tempb; tempb(2);10]); 
gam = tmpt(1:3);
nu = tmpt(4);
while sum(exp(gam(2:3))) > .999 
    gam(2:3) = log(.99*exp(gam(2:3)));
end
ybar = mean(y);
s2 = var(y)/T;
countgam = 0;
countmu = 0;

disp('Starting GARCH-t.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
        %% sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec sig2c] = loglike_garch_t(y-muc,gam,sig20,nu);
    [llike sig2] = loglike_garch_t(y-mu,gam,sig20,nu);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
    
        %% sample gam      
    e = y-mu;
    [llike sig2] = loglike_garch_t(e,gam,sig20,nu);
    if loop == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        tmpt = fminsearch(@(x)-loglike_garch_t(e,x(1:3),sig20,x(4))...
            -lpri_gam(x(1:3))-lpri_nu(x(4)),tmpt);        
        gamt = tmpt(1:3);
        nut = tmpt(4);         
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
        S = [repmat(.5./sig2.*(e2./(sig2*nut)./(1+e2./sig2/nut)-1),1,3) .* dsig2 ...
            -.5*log(1+e2./sig2/nut) + (nut+1)/(2*nut^2)*e2./sig2./(1+e2./sig2/nut)];
        I = S'*S;        
        lprop = @(x) -.5*(x-tmpt)'*I*(x-tmpt);
        Ctmp = chol(I,'lower');
    end    
    tmpc = tmpt + Ctmp'\randn(4,1);
    gamc = tmpc(1:3);    
    nuc = tmpc(4);    
    if sum(exp(gamc(2:3))) < .999 %% impose stationarity
        [llikec sig2c] = loglike_garch_t(e,gamc,sig20,nuc);
        alpMH = llikec + lpri_gam(gamc) + lpri_nu(nuc) - lprop([gamc;nuc])...
            - (llike + lpri_gam(gam) + lpri_nu(nu) - lprop([gam; nu]));
        if alpMH > log(rand)
            nu = nuc;
            gam = gamc;
            sig2 = sig2c;
            llike = llikec;
            countgam = countgam + 1;        
        end
    end
    if loop>burnin
        i = loop-burnin;        
        store_theta(i,:) = [mu exp(gam)' nu];
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
    [ml mlstd] = ml_garch_t(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('nu          | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end



