% =======================================================================
% This script estimates the UC-MA model.
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models 
% with Application to Inflation Forecast, Journal of Econometrics, 
% 176 (2), 162-172.
% =======================================================================

%% prior
tau0 = 0; invVtau = 1/5;
phih0 = .9; invVphih = 1;
muh0 = 0; invVmuh = 1/5;
invVpsi = 1; 
nutau = 10; Stau = .02*(nutau-1);
nuh = 10; Sh = .05*(nuh-1);

disp('Starting UC-MA.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigtau2 = .05;
sigh2 = .05;
phih = .9;
muh = 1;
invsigtau2 = 1/sigtau2;
invDpsic = .01;
h = log(var(y)*.8)*ones(T,1);
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
psi = 0;
psihat = psi; 
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
countpsi = 0;
countphih = 0;

% initialize for storage
stheta = zeros(nloop - burnin,4); % [muh phih sigh2 sigtau2]
spsi = zeros(nloop - burnin,1); 
stau = zeros(nloop - burnin,T); 
sh = zeros(nloop - burnin,T);

%% compute a few things outside the loop
newnutau = (T-1)/2 + nutau;
newnuh = T/2 + nuh;
psipri = @(x) -log(normpdf(x,0,sqrt(1/invVpsi))/(normcdf(sqrt(invVpsi))-normcdf(-sqrt(invVpsi))));

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop

    %% sample tau    
    invS_tau = sparse(1:T,1:T,[invVtau invsigtau2*ones(1,T-1)],T,T);
    invOmega_tau = H'*invS_tau*H;  
    invS_y = sparse(1:T,1:T,exp(-h));
    invD_tautilde = invS_y + Hpsi'*invOmega_tau*Hpsi;
    C_tautilde = chol(invD_tautilde,'lower');
    tauhat = C_tautilde'\(C_tautilde\(invS_y*(Hpsi\y)));
    tautilde = tauhat + C_tautilde'\randn(T,1);
    tau = Hpsi*tautilde; 
    
    %% sample h
    Ystar = log((Hpsi\(y-tau)).^2 + .0001);
    h = SV(Ystar,h,phih,sigh2,muh);
   
    %% sample sigtau2
    newStau = Stau + sum((tau(2:end)-tau(1:end-1)).^2)/2;
    invsigtau2 = gamrnd(newnutau, 1/newStau);
    sigtau2 = 1/invsigtau2;   
    
    %% sample sigh2
    newSh = Sh + sum([(h(1)-muh)*sqrt(1-phih^2); h(2:end)-phih*h(1:end-1)-muh*(1-phih)].^2)/2;
    invsigh2 = gamrnd(newnuh, 1/newSh);
    sigh2 = 1/invsigh2;
    
    %% sample phih
    Xphih = h(1:end-1)-muh;
    yphih = h(2:end) - muh;
    Dphih = 1/(invVphih + Xphih'*Xphih/sigh2);
    phihhat = Dphih*(invVphih*phih0 + Xphih'*yphih/sigh2);
    phihc = phihhat + sqrt(Dphih)*randn;
    g = @(x) -.5*log(sigh2./(1-x.^2))-.5*(1-x.^2)/sigh2*(h(1)-muh)^2;
    if abs(phihc)<.9999
        alp = exp(g(phihc)-g(phih));
        if alp>rand
            phih = phihc;
            countphih = countphih+1;
        end
    end 
    
    %% sample muh    
    Dmuh = 1/(invVmuh + ((T-1)*(1-phih)^2 + (1-phih^2))/sigh2);
    muhhat = Dmuh*(invVmuh*muh0 + (1-phih^2)/sigh2*h(1) + (1-phih)/sigh2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;    
     
    %% sample psi
    fpsi = @(x) fMA1(x,y-tau,h) + psipri(x);
    [psi flag psihat invDpsic] = sample_psi(psi,fpsi,loop,invDpsic,options);
    Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
    countpsi = countpsi + flag;
    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end     
    
    if loop>burnin
        i = loop-burnin;
        stau(i,:) = tau';
        spsi(i,:) = psi;
        sh(i,:) = h'; 
        stheta(i,:) = [muh phih sigh2 sigtau2];
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

tauhat = mean(stau)'; 
tauCI = quantile(stau,[.05 .95])';
hhat = mean(exp(sh/2))';
hCI = quantile(exp(sh/2),[.05 .95])';

figure;
subplot(2,1,1); 
hold on 
    plotCI(tid,tauCI(:,1),tauCI(:,2));
    plot(tid,tauhat); 
    title('\tau_t'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off        
subplot(2,1,2); 
hold on 
    plotCI(tid,hCI(:,1),hCI(:,2));
    plot(tid,hhat); 
    title('exp(h_t/2)'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off

figure; 
hist(spsi,50);
title('\psi_1'); box off;

