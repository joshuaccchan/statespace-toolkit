% This script estimates an UCSV model with gamma priors on the error
% variances in the two SV state equations.
% 
% See:
% Chan, J.C.C. (2018). Specification Tests for Time-Varying Parameter 
% Models with Stochastic Volatility, Econometric Reviews, 37(8), 807-823

valh = 0;
valg = 0;

%% prior
tau0 = 0; Vtau = 10; Vh = 10; Vg = 10;
b0 = 0; Vh0 = 10;
c0 = 0; Vg0 = 10; 
Vomegah = .2;
Vomegag = .2;
    
% initialize the Markov chain
T = length(y);
h0 = log(var(y))/2;
g0 = log(var(y))/2;
omegah2 = .2;
omegag2 = .2;
omegah = sqrt(omegah2);
omegag = sqrt(omegag2);
htilde = zeros(T,1);
gtilde = zeros(T,1);
h = h0 + sqrt(omegah2)*htilde;
g = g0 + sqrt(omegah2)*gtilde; 
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);

% define a few things
npts = 500; % number of grid points
omh_grid = linspace(-1,1,npts)';
omg_grid = linspace(-1,1,npts)';

% initialize for storeage
store_theta = zeros(nloop - burnin,4); % [omegah2 omegag2 h0 g0]
store_tau = zeros(nloop - burnin,T); 
store_h = zeros(nloop - burnin,T);
store_g = zeros(nloop - burnin,T);
store_pomh = zeros(npts,1);
store_pomg = zeros(npts,1);
store_lpden = zeros(nloop - burnin,3); % log posterior densities of 
                                       % omegah = valh; omegag = valg;
                                       % omegah = omegag                  

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );    
for loop = 1:nloop
    
    %% sample tau
    HinvStauH = H'*sparse(1:T,1:T,[1/Vtau*exp(-g(1)); exp(-g(2:end))])*H;
    invDtau =  HinvStauH + sparse(1:T,1:T,1./exp(h));
    alptau = H\sparse(1,1,tau0,T,1);
    tauhat = invDtau\(HinvStauH*alptau + y./exp(h));
    tau = tauhat + chol(invDtau,'lower')'\randn(T,1);     
    
    %% sample htilde 
    Ystar = log((y-tau).^2 + .0001);
    [htilde h0 omegah omegahhat Domegah] = SVRW_gam(Ystar,htilde,h0,omegah,b0,Vh0,Vh,Vomegah); 
    h = h0 + omegah*htilde;
    omegah2 = omegah^2;
    
    %% sample g
    Ystar = log([(tau(1)-tau0)/sqrt(Vtau);tau(2:end)-tau(1:end-1)].^2 + .0001);
    [gtilde g0 omegag omegaghat Domegag] = SVRW_gam(Ystar,gtilde,g0,omegag,c0,Vg0,Vg,Vomegag); 
    g = g0 + omegag*gtilde;
    omegag2 = omegag^2; 
    
    if ( mod( loop, 20000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end     
    
    if loop>burnin
        i = loop-burnin;
        store_tau(i,:) = tau';
        store_h(i,:) = h'; 
        store_g(i,:) = g'; 
        store_theta(i,:) = [omegah2 omegag2 h0 g0]; 

        lh0 = -.5*log(2*pi*Domegah) - .5*(omegahhat-valh)^2/Domegah;
        lg0 = -.5*log(2*pi*Domegag) - .5*(omegaghat-valg)^2/Domegag;
        lhg0 = lh0 + lg0;
        store_lpden(i,:) = [lh0 lg0 lhg0];
        
        store_pomh = store_pomh + normpdf(omh_grid,omegahhat,sqrt(Domegah));
        store_pomg = store_pomg + normpdf(omg_grid,omegaghat,sqrt(Domegag));        
    end    
end

thetahat = mean(store_theta)';
thetastd = std(store_theta)';
tauhat = mean(store_tau)';
hhat = mean(exp(store_h/2))'; 
hCI = quantile(exp(store_h/2),[.05 .95])';
ghat = mean(exp(store_g/2))'; 
gCI = quantile(exp(store_g/2),[.05 .95])';
pomhhat = store_pomh/(nloop-burnin);
pomghat = store_pomg/(nloop-burnin);
priden_omh = normpdf(omh_grid,0,sqrt(Vomegah));
priden_omg = normpdf(omg_grid,0,sqrt(Vomegag));
maxlpden = max(store_lpden);
lpostden = log(mean(exp(store_lpden-repmat(maxlpden,nloop-burnin,1)))) + maxlpden;
lpriden = [log(normpdf(valh,0,sqrt(Vomegah))) log(normpdf(valg,0,sqrt(Vomegag)))];
lpriden(3) = sum(lpriden(1:2));
lBF = lpriden-lpostden;