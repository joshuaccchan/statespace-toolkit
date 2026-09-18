% This script estimates the UC model with orthogonal errors.
%
% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

%% prior
mu0 = .75; Vmu = 1^2;
tau00 = 750; Vtau0 = 100;
phi0 = [1.3 -.7]'; invVphi = speye(2);
sigc2_ub = 3;
sigtau2_ub = 3;
pri_sigc2 = @(x) log(1/sigc2_ub);
pri_sigtau2 = @(x) log(1/sigtau2_ub);
R = 50000;
count = 0;
tempphi = repmat(phi0',R,1) + (chol(invVphi\speye(2),'lower')*randn(2,R))';
for i=1:R
    phic = tempphi(i,:)';
    if sum(phic) < .99 && phic(2) - phic(1) < .99 && phic(2) > -.99
        count = count+1;
    end    
end
phi_const = 1/(count/R);
prior = @(m,ph,sy,st,t0) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
    -log(2*pi)+.5*log(det(invVphi))+log(phi_const)-.5*(ph-phi0)'*invVphi*(ph-phi0)...
    + pri_sigc2(sy) + pri_sigtau2(st) ...    
    -.5*log(2*pi*Vtau0) - .5*(t0-tau00)^2/Vtau0;

disp('Starting MCMC for UC0.... ');
disp(' ' );
start_time = clock; 

% initialize the Markov chain
mu = .81;
phi = [1.34 -.7]';
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
HH = H'*H;
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
    - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
tau0 = y(1);
sigc2 = .5;
sigtau2 = 1.5;
rho = 0;
ngrid = 500;

%% compute a few things
Xbeta = [[1;sparse(T-1,1)] ones(T,1)];
XbetaXbeta = Xbeta'*Xbeta;

%% initialize for storeage
store_theta = zeros(nsims,6); % [mu, phi, sigc2, sigtau2, tau0]
store_tau = zeros(nsims,T); 
countphi = 0;

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for isim = 1:nsims+burnin
     
    %% sample tau  
    alp = H\(mu+[tau0;sparse(T-1,1)]);
    a = -rho*sqrt(sigc2/sigtau2)*(H*alp);
    B = Hphi + rho*sqrt(sigc2/sigtau2)*H;
    tmpc = 1/((1-rho^2)*sigc2);
    Ktau = HH/sigtau2 + tmpc*(B'*B);
    tauhat = Ktau\(HH*alp/sigtau2 + tmpc*B'*(Hphi*y-a));
    tau = tauhat + chol(Ktau,'lower')'\randn(T,1);

    %% sample phi
    e = y-tau;
    Xphi = [[0;e(1:T-1)] [0;0;e(1:T-2)]];
    tmpc = 1/((1-rho^2)*sigc2);
    Kphi = invVphi + tmpc*(Xphi'*Xphi);
    phihat = Kphi\(invVphi*phi0 ... 
        + tmpc*Xphi'*(e-rho*sqrt(sigc2/sigtau2)*H*(tau-alp)));
    flag = 0; count = 0;
    while flag == 0 && count < 100
        phic = phihat + chol(Kphi,'lower')'\randn(2,1);
        if sum(phic) < .99 && phic(2) - phic(1) < .99 && phic(2) > -.99
            phi = phic;
            flag = 1;
            countphi = countphi + 1;
        end
        count = count + 1;
    end
    Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
        - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);    
        
    %% sample sigc2
    u = [Hphi*(y-tau) (tau-[tau0; tau(1:end-1)])-mu];
    c1 = sum(u(:,1).^2);
    c2 = u(:,1)'*u(:,2); 
    c3 = sum(u(:,2).^2);
    gy = @(x) -T/2*log(x) - 1./(2*(1-rho^2)*x).*(c1-2*rho*sqrt(x/sigtau2)*c2...
        + rho^2*x/sigtau2*c3);
    sigc2grid = linspace(.02-rand/100,sigc2_ub+rand/100,ngrid);
    logpsigc2 = gy(sigc2grid) + pri_sigc2(sigc2grid);    
    psigc2 = exp(logpsigc2-max(logpsigc2));
    psigc2 = psigc2/sum(psigc2);
    cumsumy = cumsum(psigc2);
    sigc2 = sigc2grid(find(rand<cumsumy, 1 ));
    
    %% sample sigtau2    
    gtau = @(x) -T/2*log(x) - c3./(2*x) ...
        - 1/(2*(1-rho^2)*sigc2)*(-2*rho*sqrt(sigc2./x)*c2 + rho^2*sigc2./x*c3);
    sigtau2grid = linspace(.02-rand/100,sigtau2_ub+rand/100,ngrid);
    logpsigtau2 = gtau(sigtau2grid) + pri_sigtau2(sigtau2grid);
    psigtau2 = exp(logpsigtau2-max(logpsigtau2));
    psigtau2 = psigtau2/sum(psigtau2);
    cumsumtau = cumsum(psigtau2);
    sigtau2 = sigtau2grid(find(rand<cumsumtau, 1 ));    
    
    %% sample mu and tau0
    uc = Hphi*(y-tau);
    Xdel = [ones(T,1) H\ones(T,1)];    
    Kdel = diag([1/Vtau0  1/Vmu]) + Xdel'*HH*Xdel/((1-rho^2)*sigtau2);
    delhat = Kdel\([tau00/Vtau0; mu0/Vmu] + 1/((1-rho^2)*sigtau2)*Xdel'*HH*...
        (tau - rho*sqrt(sigtau2/sigc2)*(H\uc)));
    del = delhat + chol(Kdel,'lower')'\randn(2,1);
    tau0 = del(1);
    mu = del(2);    

    if ( mod( isim, 10000 ) ==0 )
        disp(  [ num2str( isim ) ' loops... ' ] )
    end     
    
    if isim>burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        store_theta(isave,:) = [mu phi' sigc2 sigtau2 tau0];
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');        
    [ml,mlstd] = ml_UC0(y,store_theta,prior,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
end

tauhat = median(store_tau)';
tauCI = quantile(store_tau,[.1 .9])';
thetahat = mean(store_theta)';
thetastd = std(store_theta)';

figure;  
subplot(1,2,1); 
hold on 
    plot(tid,tauhat, 'LineWidth',1,'Color','blue');
    plot(tid,y,'--k','LineWidth',1);
hold off
box off; xlim([tid(1)-1 tid(end)+1]);
legend('Trend', 'GDP','Location','NorthWest');    
subplot(1,2,2);
tmpy = repmat(y,1,2)-tauCI; 
hold on        
    plotCI(tid,tmpy(:,1),tmpy(:,2));
    plot(tid, (y-tauhat), 'LineWidth',1,'Color','blue');
    plot(tid, zeros(T,1),'-k','LineWidth',1);
hold off
box off; xlim([tid(1)-1 tid(end)+1]);
set(gcf,'Position',[100 100 800 300])  %% 1 by 2 graphs

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('phi_1       | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('phi_2       | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('sigma^2_c   | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('sigma^2_tau | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end