% This script estimates the GARCH(1,2) (GARCH-2) model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
mu0 = 0; Vmu = 10;
gam0 = log([exp(1) .1 .8 .1]'); Vgam = diag([10 1 1 1]);
        % estimate stationarity prob
tempgam = repmat(gam0',10000,1) + (chol(Vgam,'lower')*randn(4,10000))';
prob_s = sum(sum(exp(tempgam(:,2:end)),2)<1)/10000;
c_prior = -.5*log(2*pi*Vmu) -2*log(2*pi) -.5*log(det(Vgam)) -log(prob_s);
lpri_gam = @(x) -.5*(x-gam0)'*(Vgam\(x-gam0));
lpri_mu = @(x) -.5*(x-mu0)^2/Vmu;
prior = @(m,g) c_prior + lpri_mu(m) + lpri_gam(g);

    %% initialize for storeage
store_theta = zeros(nloop - burnin,5); % mu, gam
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
gamt = fminsearch(@(x)-loglike_garch_2(e,x,sig20) -lpri_gam(x),...
    [tempb; tempb(2); tempb(2)/2]);
expgamt = exp(gamt);
sig2 = zeros(T,1);
dsig2 = zeros(T,4);
dsig2(1,1) = expgamt(1);
dsig2(1,3) = expgamt(3)*sig20;
sig2(1) = expgamt(1) + expgamt(3)*sig20;            
for t=2:T
    if t==2
        sig2(t) = expgamt(1) + expgamt(2)*e2(1) ...
            + expgamt(3)*sig2(1) + expgamt(4)*sig20;
        dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(1,1);
        dsig2(t,2) = expgamt(2)*e2(1) + expgamt(3)*dsig2(1,2);
        dsig2(t,3) = expgamt(3)*(sig2(1) + dsig2(1,3));
        dsig2(t,4) = expgamt(3)*dsig2(1,4) + expgamt(4)*sig20;
    else
        sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) ...
            + expgamt(3)*sig2(t-1) + expgamt(4)*sig2(t-2);
        dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1)...
            + expgamt(4)*dsig2(t-2,1);
        dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2) ...
            + expgamt(4)*dsig2(t-2,2);
        dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3)) ...
            + expgamt(4)*dsig2(t-2,3);
        dsig2(t,4) = expgamt(3)*dsig2(t-1,4) ...
            + expgamt(4)*(sig2(t-2) + dsig2(t-2,4));
    end
end
S = repmat(.5./sig2.*(e2./sig2-1),1,4) .* dsig2;
I = S'*S + Vgam\speye(4);
Cgam = chol(I,'lower');     
gam = gamt + Cgam'\randn(4,1);    
while sum(exp(gam(2:end))) > .999 
    gam = gamt + Cgam'\randn(4,1);    
end
ybar = mean(y);
s2 = var(y)/T;
countgam = 0;
countmu = 0;

disp('Starting GARCH-2.... ');
disp(' ' );
randn('seed',sum(clock*97)); rand('seed',sum(clock*37));

start_time = clock;
for loop = 1:nloop
        %% sample mu
    muc = ybar + sqrt(s2)*randn;
    [llikec sig2c] = loglike_garch_2(y-muc,gam,sig20);
    [llike sig2] = loglike_garch_2(y-mu,gam,sig20);
    alpMH = llikec + lpri_mu(muc) + .5*(muc-ybar)^2/s2 ...
        - (llike + lpri_mu(mu) + .5*(mu-ybar)^2/s2);
    if alpMH > log(rand)
        mu = muc;
        countmu = countmu + 1;        
    end
    
        %% sample gam      
    e = y-mu;
    [llike sig2] = loglike_garch_2(e,gam,sig20);    
    if loop == 1 || rand>.5  %% maximize with prob 0.5
        e2 = e.^2;        
        gamt = fminsearch(@(x) -loglike_garch_2(e,x,sig20)-lpri_gam(x),gamt);
        expgamt = exp(gamt);
        sig2 = zeros(T,1);
        dsig2 = zeros(T,4);
        dsig2(1,1) = expgamt(1);
        dsig2(1,3) = expgamt(3)*sig20;
        sig2(1) = expgamt(1) + expgamt(3)*sig20;            
        for t=2:T
            if t==2
                sig2(t) = expgamt(1) + expgamt(2)*e2(1) ...
                    + expgamt(3)*sig2(1) + expgamt(4)*sig20;
                dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(1,1);
                dsig2(t,2) = expgamt(2)*e2(1) + expgamt(3)*dsig2(1,2);
                dsig2(t,3) = expgamt(3)*(sig2(1) + dsig2(1,3));
                dsig2(t,4) = expgamt(3)*dsig2(1,4) + expgamt(4)*sig20;
            else
                sig2(t) = expgamt(1) + expgamt(2)*e2(t-1) ...
                    + expgamt(3)*sig2(t-1) + expgamt(4)*sig2(t-2);
                dsig2(t,1) = expgamt(1) + expgamt(3)*dsig2(t-1,1)...
                    + expgamt(4)*dsig2(t-2,1);
                dsig2(t,2) = expgamt(2)*e2(t-1) + expgamt(3)*dsig2(t-1,2) ...
                    + expgamt(4)*dsig2(t-2,2);
                dsig2(t,3) = expgamt(3)*(sig2(t-1) + dsig2(t-1,3)) ...
                    + expgamt(4)*dsig2(t-2,3);
                dsig2(t,4) = expgamt(3)*dsig2(t-1,4) ...
                    + expgamt(4)*(sig2(t-2) + dsig2(t-2,4));
            end
        end
        S = repmat(.5./sig2.*(e2./sig2-1),1,4) .* dsig2;
        I = S'*S  + Vgam\speye(4);    
        lprop = @(x) -.5*(x-gamt)'*I*(x-gamt);
        Cgam = chol(I,'lower');
    end
    count = 0;
    gamc = gamt + Cgam'\randn(4,1);
    while sum(exp(gamc(2:end))) > .999 && count < 100
        gamc = gamt + Cgam'\randn(4,1);
        count = count + 1;
    end    
    [llikec sig2c] = loglike_garch_2(e,gamc,sig20);
    alpMH = llikec + lpri_gam(gamc) - lprop(gamc)...
        - (llike + lpri_gam(gam) - lprop(gam));
    if alpMH > log(rand)
        gam = gamc;
        sig2 = sig2c;
        llike = llikec;
        countgam = countgam + 1;        
    end    
    if loop>burnin
        i = loop-burnin;        
        store_theta(i,:) = [mu exp(gam)'];
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
    [ml mlstd] = ml_garch_2(y,store_theta,prior,sig20,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] ); 
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('alpha_0     | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('alpha_1     | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('beta_1      | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('beta_2      | %.2f (%.2f)\n', thetahat(5), thetastd(5));

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end