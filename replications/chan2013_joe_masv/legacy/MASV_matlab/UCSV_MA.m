% =======================================================================
% This script estimates the UCSV-MA model.
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models 
% with Application to Inflation Forecast, Journal of Econometrics, 
% 176 (2), 162-172.
% =======================================================================

%% prior
invVpsi = 1;
phih0 = .9; invVphih = 1;
phig0 = .9; invVphig = 1;
muh0 = 0; invVmuh = 1/5;
mug0 = 0; invVmug = 1/5;
nuh0 = 10; Sh0 = .05*(nuh0-1);
nug0 = 10; Sg0 = .05*(nug0-1);

disp('Starting UCSV-MA.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigh2 = .05; sigg2 = .05;
phih = .99;  phig = .99;
muh = 1; mug = 1;
h = log(var(y)*.5)*ones(T,1);
g = log(var(y)*.5)*ones(T,1);
psihat = 0;
psi = psihat;
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
invDpsic = .01;
countpsi = 0;
countphih = 0;
countphig = 0;

%% initialize for storeage
stheta = zeros(nloop - burnin,6);  %[muh mug phih phig sigh2 sigg2]
spsi = zeros(nloop - burnin,1); 
stau = zeros(nloop - burnin,T); 
sh = zeros(nloop - burnin,T);
sg = zeros(nloop - burnin,T);

%% compute a few things outside the loop
newnuh = T/2 + nuh0;
newnug = T/2 + nug0;
psipri = @(x) -log(normpdf(x,0,sqrt(1/invVpsi))/(normcdf(sqrt(invVpsi))-normcdf(-sqrt(invVpsi))));

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for loop = 1:nloop

    %% sample tau    
    invS_tau = sparse(1:T,1:T,exp(-g));
    invOmega_tau = H'*invS_tau*H;  
    invS_y = sparse(1:T,1:T,exp(-h));
    invD_tautilde = invS_y + Hpsi'*invOmega_tau*Hpsi;
    tauhat = invD_tautilde\(invS_y*(Hpsi\y));
    tautilde = tauhat + chol(invD_tautilde)\randn(T,1);
    tau = Hpsi*tautilde;
    
    %% sample h
    Ystar = log((Hpsi\(y-tau)).^2 + .0001);
    h = SV(Ystar,h,phih,sigh2,muh);
   
    %% sample g
    Ystar = log([tau(1); tau(2:end)-tau(1:end-1)].^2 + .0001);
    g = SV(Ystar,g,phig,sigg2,mug);   
   
    %% sample sigh2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];
    newSh = Sh0 + sum(errh.^2)/2;
    sigh2 = 1/gamrnd(newnuh, 1./newSh);     
    
    %% sample sigg2
    errg = [(g(1)-mug)*sqrt(1-phig^2);  g(2:end)-phig*g(1:end-1)-mug*(1-phig)];
    newSg = Sg0 + sum(errg.^2)/2;
    sigg2 = 1/gamrnd(newnug, 1./newSg);

    %% sample psi    
    fpsi = @(x) fMA1(x,y-tau,h) + psipri(x);
    [psi flag psihat invDpsic] = sample_psi(psi,fpsi,loop,invDpsic,options);
    Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
    countpsi = countpsi + flag;
    
    %% sample phih
    Xphih = h(1:end-1)-muh;
    yphih = h(2:end) - muh;
    Dphih = 1/(invVphih + Xphih'*Xphih/sigh2);
    phihhat = Dphih*(invVphih*phih0 + Xphih'*yphih/sigh2);
    phihc = phihhat + sqrt(Dphih)*randn;
    fh = @(x) -.5*log(sigh2./(1-x.^2))-.5*(1-x.^2)/sigh2*(h(1)-muh)^2;
    if abs(phihc)<.9999
        alp = exp(fh(phihc)-fh(phih));
        if alp>rand
            phih = phihc;
            countphih = countphih+1;
        end
    end 
    
    %% sample phig
    Xphig = g(1:end-1)-mug;
    yphig = g(2:end) - mug;
    Dphig = 1/(invVphig + Xphig'*Xphig/sigg2);
    phighat = Dphig*(invVphig*phig0 + Xphig'*yphig/sigg2);
    phigc = phighat + sqrt(Dphig)*randn;
    fg = @(x) -.5*log(sigg2./(1-x.^2))-.5*(1-x.^2)/sigg2*(g(1)-mug)^2;
    if abs(phigc)<.9999
        alp = exp(fg(phigc)-fg(phig));
        if alp>rand
            phig = phigc;
            countphig = countphig+1;
        end
    end 
    
    %% sample muh    
    Dmuh = 1/(invVmuh + ((T-1)*(1-phih)^2 + (1-phih^2))/sigh2);
    muhhat = Dmuh*(invVmuh*muh0 + (1-phih^2)/sigh2*h(1) + (1-phih)/sigh2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;   
    
    %% sample mug
    Dmug = 1/(invVmug + ((T-1)*(1-phig)^2 + (1-phig^2))/sigg2);
    mughat = Dmug*(invVmug*mug0 + (1-phig^2)/sigg2*g(1) + (1-phig)/sigg2*sum(g(2:end)-phig*g(1:end-1)));
    mug = mughat + sqrt(Dmug)*randn;   
    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
    
    if loop>burnin
        i = loop-burnin;
        stau(i,:) = tau';
        spsi(i,:) = psi;
        sh(i,:) = h'; 
        sg(i,:) = g'; 
        stheta(i,:) = [muh mug phih phig sigh2 sigg2];       
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

tauhat = mean(stau)';
tauCI = quantile(stau,[.05 .95])';
hhat = mean(exp(sh/2))'; 
hCI = quantile(exp(sh/2),[.05 .95])';
ghat = mean(exp(sg/2))'; 
gCI = quantile(exp(sg/2),[.05 .95])';

figure;
hold on 
    plotCI(tid,tauCI(:,1),tauCI(:,2));
    plot(tid,tauhat); 
    title('\tau_t'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off 

figure;
subplot(2,1,1); 
hold on 
    plotCI(tid,hCI(:,1),hCI(:,2));
    plot(tid,hhat); 
    title('exp(h_t/2)'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off
subplot(2,1,2); 
hold on 
    plotCI(tid,gCI(:,1),gCI(:,2));
    plot(tid,ghat); 
    title('exp(g_t/2)'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off

figure; 
hist(spsi,50);
title('\psi_1'); box off;



