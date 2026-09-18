% =======================================================================
% This script estimates the AR(2)-MA model.
%
% See:
% Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models 
% with Application to Inflation Forecast, Journal of Econometrics, 
% 176 (2), 162-172.
% =======================================================================

y0 = USCPI_Q(1:2); 
y = USCPI_Q(3:end); 
tid = (1947.75:.25:2011.5)';
q = 3 ;  % # of AR lags + intercept
T = T-1; 

%% prior
phih0 = .9; invVphih = 1;
muh0 = 0; invVmuh = 1/5;
nuh = 10; Sh = .05*(nuh-1);
invVbeta = ones(q,1)/5;
invVpsi = 1;

% initialize the Markov chain
X = [ones(T,1) [y0(end); y(1:end-1)] [y0; y(1:end-2)]];
psi = 0;
psihat = psi;
beta = (X'*X)\(X'*y);
muh = 0; phih = .8; sigh2 = .1;
h = log(var(y-X*beta))*ones(T,1);
invDpsic = .01;
countpsi = 0;
countphih=0;

% initialize for storage
stheta = zeros(nloop-burnin,3);
sbeta = zeros(nloop-burnin,q);
spsi = zeros(nloop-burnin,1);
sh = zeros(nloop - burnin,T);

%% construct a few things
newnuh = T/2 + nuh;
psipri = @(x) -log(normpdf(x,0,sqrt(1/invVpsi))/(normcdf(sqrt(invVpsi))-normcdf(-sqrt(invVpsi))));
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 

disp('Starting AR(2)-MA.... ');
disp(' ' );
start_time = clock;    

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop

    %% sample beta     
    Xtilde = Hpsi\X;    ytilde = Hpsi\y;
    temp1 = Xtilde'*sparse(1:T,1:T,exp(-h));
    invDbeta = temp1*Xtilde + sparse(1:q,1:q,invVbeta);      
    betahat = invDbeta\(temp1*ytilde);
    Cbeta = chol(invDbeta);
    flag = 0;
    while flag ==0
        beta = betahat + Cbeta\randn(q,1);
        if beta(3)> -.98 && beta(3) < 1-abs(beta(2)) %% stationarity conditions for AR(2)
            flag = 1;
        end
    end    
    Xbeta = X*beta;

    %% sample h    
    Ystar = log((Hpsi\(y-Xbeta)).^2 + .0001);
    h = SV(Ystar,h,phih,sigh2,muh);
    
    %% sample sigh2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];
    newSh = Sh + sum(errh.^2)/2;
    sigh2 = 1/gamrnd(newnuh, 1./newSh);
    
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
    fpsi = @(x) fMA1(x,y-Xbeta,h) + psipri(x);
    [psi flag psihat invDpsic] = sample_psi(psi,fpsi,loop,invDpsic,options);
    Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
    countpsi = countpsi + flag;
    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
    
    if loop>burnin
        i=loop-burnin;
        sbeta(i,:) = beta'; 
        sh(i,:) = h'; 
        stheta(i,:) = [muh phih sigh2];    
        spsi(i) = psi;     
    end
    
end
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

betahat = mean(sbeta);
hhat = mean(exp(sh/2))';
hCI = quantile(exp(sh/2),[.05 .95])';

figure;
hold on 
    plotCI(tid,hCI(:,1),hCI(:,2));
    plot(tid,hhat); 
    title('exp(h_t/2)'); xlim([tid(1)-.5 tid(end)+.5]); box off;
hold off

figure; 
hist(spsi,50);
title('\psi_1'); box off;