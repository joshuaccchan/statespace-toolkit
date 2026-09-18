% This script estimates a bivariate unobserved components model for 
% inflation and unemployment.
% 
% See:
% Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter 
% Models with Stochastic Volatility, Econometric Reviews, 37(8), 807-823

%% prior 
Vtau = 10; Vnu = 10; Vh = 10; Vg = 10;
a0 = 0;  Vnu0 = 10; % prior mean and variance of nu0
b0 = 0;  Vh0 = 10; % prior mean and variance of h0
c0 = 0;  Vg0 = 10; % prior mean and variance of g0
Vomeganu = .1;
Vomegah = .2;
Vomegag = .2;
phi0 = [0;0];
Vphi = eye(2); invVphi = Vphi\speye(2);
nuu = 5; Su = .1;
lam0 = 0; Vlam = 10;
tau0 = 0;
beta0 = [a0;0];
invVbeta = diag([1/Vnu0 1/Vomeganu]);

%% define a few things
npts = 500; % number of grid points
omnu_grid = linspace(-1,1,npts)';

%% initialize for storage %%
store_theta = zeros(nloop - burnin,10); 
store_tau = zeros(nloop - burnin,T); 
store_nu = zeros(nloop - burnin,T); 
store_h = zeros(nloop - burnin,T);
store_g = zeros(nloop - burnin,T);
store_lpden = zeros(nloop - burnin,1);
store_pomnu = zeros(npts,1);

%% initialize the Markov chain %%
sigu2 = var(u);
nu0 = mean(u);
omeganu2 = Vomeganu;
omeganu = sqrt(omeganu2);
nutilde = nu0 + .1*omeganu*randn(T,1);
omegatau = sqrt(.2);
phi = [.5 .2]';
h0 = log(var(y)*.5);
g0 = -1;
htilde = zeros(T,1);
gtilde = zeros(T,1);
omegag2 = Vomegag;
omegah2 = Vomegah;   
omegah = sqrt(omegah2);
omegag = sqrt(omegag2);
h = h0 + omegah*htilde;
g = g0 + omegag*gtilde;
lam = 0;

%% compute and define a few things %%
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
     - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
countpsi = 0;

randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));

for loop = 1:nloop    
  
        %% sample tau
    exph = exp(h);
    HinvStauH = H'*sparse(1:T,1:T,[1/Vtau*exp(-g(1)); exp(-g(2:end))])*H;
    Ktau =  HinvStauH + sparse(1:T,1:T,1./exph);
    alptau = H\sparse(1,1,tau0,T,1);
    tauhat = Ktau\(HinvStauH*alptau + (y - lam*(u-nu0-omeganu*nutilde))./exph);
    tau = tauhat + chol(Ktau,'lower')'\randn(T,1);         
    
        %% sample nu
    invSpi = sparse(1:T,1:T,1./exph);
    Hphi2 = Hphi'*Hphi;    
    Knutilde = lam^2*omeganu2*invSpi + omeganu2/sigu2*Hphi2 ...
        + H'*sparse(1:T,1:T,[1/Vnu ones(1,T-1)])*H;
    nutildehat = Knutilde\(omeganu/sigu2*Hphi2*(u-nu0) ...
        - lam*omeganu*invSpi*(y-tau-lam*(u-nu0)));
    nutilde = nutildehat + chol(Knutilde,'lower')'\randn(T,1);    
    
        %% sample nu0 and omeganu    
    Xbeta = [ones(T,1) nutilde];      
    XbetainvSpi = Xbeta'*invSpi;
    Kbeta = invVbeta + lam^2*XbetainvSpi*Xbeta + Xbeta'*Hphi2*Xbeta/sigu2;
    betahat = Kbeta\(invVbeta*beta0 + 1/sigu2*Xbeta'*Hphi2*u ...
        - lam*XbetainvSpi*(y-tau-lam*u));   
    beta = betahat + chol(Kbeta,'lower')'\randn(2,1);
    nu0 = beta(1); omeganu = beta(2); 
    
        % permute nutilde and omeganu
    U = -1 + 2*(rand>0.5);
    nutilde = U*nutilde;
    omeganu = U*omeganu;
    nu = nu0 + omeganu*nutilde;
    omeganu2 = omeganu^2;
    
       %% sample phi
    e = u - nu0 - omeganu*nutilde;
    Xphi = [e(2:end-1) e(1:end-2)];
    invDphi = invVphi + Xphi'*Xphi/sigu2;
    phihat = invDphi\(invVphi*phi0 + Xphi'*e(3:end)/sigu2);
    phic = phihat + chol(invDphi,'lower')'\randn(2,1);
    if sum(phic) < .999 && phic(2) - phic(1) < .999 && abs(phic(2)) < .999
        phi = phic;
        Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
            - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
    end     
        %% sample sigu2
    newSu = Su + sum((Hphi*(u-Xbeta*beta)).^2)/2;
    sigu2 = 1/gamrnd(nuu+T/2,1/newSu);
     
        %% sample lam
    Xlam = u-nu0-omeganu*nutilde;
    Dlam = 1/(1/Vlam + Xlam'*invSpi*Xlam);
    lamhat = Dlam*(lam0/Vlam + Xlam'*invSpi*(y-tau));
    lam = lamhat + sqrt(Dlam)*randn;
        
    %% sample h
    Ystar = log((y-tau-lam*(u-nu0-omeganu*nutilde)).^2 + .0001);
    [htilde h0 omegah] = SVRW_gam(Ystar,htilde,h0,omegah,b0,Vh0,Vh,Vomegah);     
    h = h0 + omegah*htilde;
    omegah2 = omegah^2;    
     
    %% sample g
    Ystar = log([(tau(1)-tau0)/sqrt(Vtau);tau(2:end)-tau(1:end-1)].^2 + .0001);
    [gtilde g0 omegag] = SVRW_gam(Ystar,gtilde,g0,omegag,c0,Vg0,Vg,Vomegag); 
    g = g0 + omegag*gtilde;
    omegag2 = omegag^2;
 
    if loop>burnin
        i = loop-burnin;
        store_tau(i,:) = tau';
        store_nu(i,:) = nu';
        store_h(i,:) = h'; 
        store_g(i,:) = g'; 
        store_theta(i,:) = [lam; phi; sigu2; omegah2; omegag2; omeganu2; ...
            nu0; h0; g0]';
         
        %% evaluate the conditional distribution of omeganu at 0
        Xbeta = [ones(T,1) nutilde];      
        XbetainvSpi = Xbeta'*invSpi;
        Dbeta = (invVbeta + lam^2*XbetainvSpi*Xbeta ...
            + Xbeta'*Hphi2*Xbeta/sigu2)\speye(2);
        betahat = Dbeta*(invVbeta*beta0 + 1/sigu2*Xbeta'*Hphi2*u ...
            - lam*XbetainvSpi*(y - tau - lam*u));
        lnu0 = -.5*log(2*pi*Dbeta(2,2)) - .5*betahat(2)^2/Dbeta(2,2);
         
        store_lpden(i,:) = lnu0;
        store_pomnu = store_pomnu + normpdf(omnu_grid,betahat(2),sqrt(Dbeta(2,2)));
    end
    
    if ( mod( loop, 20000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end 
    
end

tauhat = mean(store_tau)';
nuhat = mean(store_nu)';
hhat = mean(store_h)';
ghat = mean(store_g)';
thetahat = mean(store_theta)';
thetastd = std(store_theta)';
thetaCI = quantile(store_theta,[.05 .95])';
nuCI = quantile(store_nu,[.05 .95])';
pomnuhat = store_pomnu/(nloop-burnin);
priden_omnu = normpdf(omnu_grid,0,sqrt(Vomeganu));

maxlpden = max(store_lpden);
lpostden = log(mean(exp(store_lpden-repmat(maxlpden,nloop-burnin,1)))) + maxlpden;
lpriden = -.5*log(2*pi*Vomeganu);
lBF = lpriden - lpostden;