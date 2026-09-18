% This script estimates the bivariate UC model with correlated errors and
% one break.

% See:
% Grant, A.L. and Chan, J.C.C. (2017). A Bayesian Model Comparison for 
% Trend-Cycle Decompositions of Output, Journal of Money, Credit and Banking,
% 49(2-3): 525-552

%% prior
mu0 = [.75 0]'; Vmu = [1 1]';
tau00 = [750 10]'; Vtau0 = [100 100]';
phi0 = [1.3 1.3 -.7 -.7]'; iVphi = speye(4);
nu0 = 7; S0 = eye(4);

R = 50000;
count = 0;
tempphi = repmat(phi0(1:2)',R,1) + (chol(iVphi(1:2,1:2)\speye(2),'lower')*randn(2,R))';
for i=1:R
    phic = tempphi(i,:)';
    if sum(phic) < .99 && phic(2) - phic(1) < .99 && phic(2) > -.99
        count = count+1;
    end    
end
phi_const = 1/(count/R);
prior = @(m,ph,ta0,S) -4/2*log(2*pi)-2/2*sum(log(Vmu))-.5*sum((m-repmat(mu0,2,1)).^2./repmat(Vmu,2,1)) ...
    -4/2*log(2*pi)+.5*log(det(iVphi))+2*log(phi_const)-.5*(ph-phi0)'*iVphi*(ph-phi0)...
    -2/2*log(2*pi)-.5*sum(log(Vtau0))-.5*sum((ta0-tau00).^2./Vtau0) ...
    +linvwishpdf(S,nu0,S0);     
 
 
disp('Starting MCMC for biUCUR-t0.... ');
disp(' ' );
start_time = clock; 
 
% initialize the Markov chain
phi = [1.35 1.35 -.7 -.7]';
mu = [.75 0 .5 0]';
Sig = eye(4);
H = speye(2*T) - sparse(3:T*2,1:(T-1)*2,ones(1,2*(T-1)),2*T,2*T);
Hphi = speye(2*T) - sparse(3:T*2,1:(T-1)*2,repmat(phi(1:2),1,T-1),2*T,2*T) ...
    -sparse(5:T*2,1:(T-2)*2,repmat(phi(3:4),1,T-2),2*T,2*T);
tau0 = y(1:2);

%% compute a few things
d0 = ((1:T)'<t0);
d1 = ((1:T)'>=t0);

%% initialize for storeage
store_theta = zeros(nsims,20); % [mu, phi, tau0, Sig]
store_tau = zeros(nsims,2*T); 
countphi = 0;

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for isim = 1:nsims+burnin
      
    %% sample tau 
    alp = H\(kron(d0,mu(1:2))+ kron(d1,mu(3:4)) +[tau0;sparse(2*(T-1),1)]);
    tmp1 = Sig(1:2,3:4)/Sig(3:4,3:4);
    tmp2 = Sig(1:2,1:2) - tmp1*Sig(1:2,3:4)';
    a = -kron(speye(T),tmp1)*(H*alp);
    B = Hphi + kron(speye(T),tmp1)*H;
    HiStauH = H'*(kron(speye(T),Sig(3:4,3:4)\speye(2)))*H;
    BiScgtau = B'*kron(speye(T),tmp2\speye(2));
    Ktau = HiStauH + BiScgtau*B;
    tauhat = Ktau\(HiStauH*alp + BiScgtau*(Hphi*y-a));
    tau = tauhat + chol(Ktau,'lower')'\randn(2*T,1);

    %% sample phi
    c = reshape(y-tau,2,T)';
    Xphi = zeros(2*T,4);
    Xphi(3:2:end,1) = c(1:T-1,1);
    Xphi(4:2:end,2) = c(1:T-1,2);
    Xphi(5:2:end,3) = c(1:T-2,1);
    Xphi(6:2:end,4) = c(1:T-2,2);
    Kphi = iVphi + Xphi'*kron(speye(T),tmp2\speye(2))*Xphi;
    phihat = Kphi\(iVphi*phi0 + Xphi'*kron(speye(T),tmp2\speye(2)) ...
        *(reshape(c',2*T,1)-kron(speye(T),tmp1)*H*(tau-alp)));
    flag = 0; count = 0;
    while flag == 0 && count < 100
        phic = phihat + chol(Kphi,'lower')'\randn(4,1);
        if (sum(phic([1 3])) < .99 && phic(3) - phic(1) < .99 && phic(3) > -.99) && ...
            (sum(phic([2 4])) < .99 && phic(4) - phic(2) < .99 && phic(4) > -.99)
            phi = phic;
            flag = 1;
            countphi = countphi + 1;
        end
        count = count + 1;
    end
    Hphi = speye(2*T) - sparse(3:T*2,1:(T-1)*2,repmat(phi(1:2),1,T-1),2*T,2*T) ...
        -sparse(5:T*2,1:(T-2)*2,repmat(phi(3:4),1,T-2),2*T,2*T);

    %% sample tau0, mu1 and mu2    
    uc = Hphi*(y-tau);
    Xdel = [kron(ones(T,1),speye(2)) H\kron(d0,speye(2)) H\kron(d1,speye(2))]; 
    tmp3 = Sig(3:4,3:4) - (Sig(1:2,3:4)'/Sig(1:2,1:2))*Sig(1:2,3:4);
    XHitmp3 = Xdel'*H'*kron(speye(T),tmp3\speye(2));    
    Kdel = diag([1./Vtau0; 1./Vmu; 1./Vmu]) + XHitmp3*H*Xdel;        
    delhat = Kdel\([tau00./Vtau0; mu0./Vmu; mu0./Vmu] + XHitmp3*...        
        H*(tau - H\(kron(speye(T),Sig(1:2,3:4)'/Sig(1:2,1:2))*uc)));    
    del = delhat + chol(Kdel,'lower')'\randn(6,1);
    tau0 = del(1:2);
    mu = del(3:end);
    
    %% sample Sig
    utau = H*tau - (kron(d0,mu(1:2))+ kron(d1,mu(3:4)) +[tau0;sparse(2*(T-1),1)]);
    u = [reshape(uc,2,T)' reshape(utau,2,T)'];
    Sig = iwishrnd(S0+u'*u,nu0+T);
 
     if ( mod( isim, 10000 ) ==0 )
         disp(  [ num2str( isim ) ' loops... ' ] )
     end     
     

    if isim > burnin
        isave = isim - burnin;
        store_tau(isave,:) = tau';
        corSig = Sig./sqrt(kron(diag(Sig),diag(Sig)'));
        store_theta(isave,:) = [mu; phi; tau0; diag(Sig); corSig([3 8 2 4 7 12])'];
    end    
end
 
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );
 
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');        
    [ml,mlstd] = ml_biUCUR_break(y,store_theta,t0,prior,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
end

tauhat = median(store_tau(:,1:2:end))';
tauCI = quantile(store_tau(:,1:2:end),[.1 .9])';
thetahat = mean(store_theta)';
thetastd = std(store_theta)';

%% plot of graphs
figure;  
subplot(1,2,1); 
hold on 
    plot(tid,tauhat, 'LineWidth',1,'Color','blue');
    plot(tid,y(1:2:end),'--k','LineWidth',1);
hold off
box off; xlim([tid(1)-1 tid(end)+1]);
legend('Trend', 'GDP','Location','NorthWest');    
subplot(1,2,2);
tmpy = repmat(y(1:2:end),1,2)-tauCI; 
hold on        
    plotCI(tid,tmpy(:,1),tmpy(:,2));
    plot(tid, (y(1:2:end)-tauhat), 'LineWidth',1,'Color','blue');
    plot(tid, zeros(T,1),'-k','LineWidth',1);
hold off
box off; xlim([tid(1)-1 tid(end)+1]); ylim([-10 8]);
set(gcf,'Position',[100 100 800 300])  %% 1 by 2 graphs

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu_1        | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_2        | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('phi_1       | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('phi_2       | %.2f (%.2f)\n', thetahat(6), thetastd(6)); 
fprintf('sigma^2_c   | %.2f (%.2f)\n', thetahat(11), thetastd(11)); 
fprintf('sigma^2_tau | %.2f (%.2f)\n', thetahat(13), thetastd(13)); 
fprintf('rho         | %.2f (%.2f)\n', thetahat(15), thetastd(15)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end 



