% This script estimates a SV model with a jump component (SV-J)
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
phih0 = .97; Vphih = .1^2;
mu0 = 0; Vmu = 10;
muh0 = 1; Vmuh = 10;
nuh = 5; Sh = .2^2*(nuh-1);
delta0 = [0 log(10)]'; Vdelta = diag([10 1]);

phih_const = 1/(normcdf(1,phih0,sqrt(Vphih))-normcdf(-1,phih0,sqrt(Vphih)));
lpri_kappa = @(x) log(1/(.1-0)) - 10^(10)*(x>.1);
lpri_delta = @(x) - log(2*pi) -.5*log(det(Vdelta)) ...
    -.5*(x-delta0)'*(Vdelta\(x-delta0));
prior = @(m,mh,ph,oh,k,d) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
    -.5*log(2*pi*Vmuh) - .5*(mh-muh0)^2/Vmuh ...
    -.5*log(2*pi*Vphih) + log(phih_const) -.5*(ph-phih0)^2/Vphih ...
    + nuh*log(Sh) - gammaln(nuh) - (nuh+1)*log(oh) - Sh/oh ...
    + lpri_delta(d) + lpri_kappa(k);

    %% initialize the Markov chain
mu = mean(y);
muh = log(var(y)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(T,1);
exph = exp(h);
sqrtexph = sqrt(exph);
kappa = .05;
q = binornd(1,kappa,T,1);
lpdelta = @(d) lpri_delta(d) -.5*sum(log(exp(d(2))*q+exp(h))) ...
    -.5*sum((y-mu-d(1)*q).^2./(exp(d(2))*q+exp(h)));
delta = fminsearch(@(d)-lpdelta(d),zeros(2,1));
k = delta(1) + exp(delta(2)/2)*randn(T,1);
    %% initialize for storage
store_theta = zeros((nloop - burnin),7); % [mu muh phih omegah2 kappa muk sigk2]
store_h = zeros((nloop - burnin),T);
store_Q = zeros(nloop - burnin,2);
    %% compute a few things outside the loop
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
newnuh = T/2 + nuh;
s2 = var(y)/T;
del_std = diag([sqrt(s2) 1]);

counth = 0; countphi = 0; countdelta = 0; countkappa = 0;
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000)); 

disp('Starting SV-J.... ');
disp(' ' );

start_time = clock;

for loop = 1:nloop 
        %% sample mu    
    invDmu = 1/Vmu + sum(1./exph);
    muhat = invDmu\sum((y-q.*k)./exph);   
    mu = muhat + chol(invDmu,'lower')'\randn;     
        %% sample q 
    p1 = kappa*normpdf(y,mu+k,sqrtexph);
    p0 = (1-kappa)*normpdf(y,mu,sqrtexph);
    q = binornd(1,p1./(p0+p1));    
        %% sample kappa
    sumq = sum(q);
    kappac = betarnd(1+sumq,1+T-sumq);
    if kappac<.1
        kappa = kappac;
        countkappa = countkappa+1;
    end    
        %% sample delta (marginal of k)
    lpdelta = @(d) lpri_delta(d) + ...
        -.5*sum(log(exp(d(2))*q+exph))-.5*sum((y-mu-d(1)*q).^2./(exp(d(2))*q+exph));
    deltahat = fminsearch(@(d)-lpdelta(d),delta);
    deltac = deltahat + del_std*randn(2,1);    
    alpMH = lpdelta(deltac) - lpdelta(delta) + ...
            -.5*sum((del_std\(delta-deltahat)).^2) + .5*sum((del_std\(deltac-deltahat)).^2);
    if alpMH > log(rand)
        delta = deltac;
        countdelta = countdelta + 1;
    end  
        
        %% sample k
    id0 = find(q==0);
    id1 = find(q==1);
    n0 = length(id0);
    Dk1 = 1./(1/exp(delta(2)) + 1./exph(id1));
    k1hat = Dk1 .* (delta(1)/exp(delta(2)) + (y(id1)-mu)./exph(id1));    
    k(id0) = delta(1) + exp(delta(2)/2)*randn(n0,1);    
    k(id1) = k1hat + sqrt(Dk1).*randn(T-n0,1);
    
        %% sample h    
    HiSH = Hphi'*sparse(1:T,1:T,[(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)])*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
    HiSHdeltah = HiSH*deltah;
    s2 = (y-mu-q.*k).^2;
    errh = 1; ht = h;
    while errh> 10^(-3);
        expht = exp(ht);
        sinvexpht = s2./expht;        
        fh = -.5 + .5*sinvexpht;
        Gh = .5*sinvexpht;        
        Kh = HiSH + sparse(1:T,1:T,Gh);
        newht = Kh\(fh+Gh.*ht+HiSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Kh,'lower');
    % AR-step:         
    lph = @(x) -.5*(x-deltah)'*HiSH*(x-deltah) -.5*sum(x) -.5*exp(-x)'*s2;
    logc = lph(ht) + log(3);   
    flag = 0;
    while flag == 0
        hc = ht + cholHh'\randn(T,1);        
        alpARc =  lph(hc) + .5*(hc-ht)'*Kh*(hc-ht) - logc;
        if alpARc > log(rand)
            flag = 1;
        end
    end        
    % MH-step    
    alpAR = lph(h) + .5*(h-ht)'*Kh*(h-ht) - logc;
    if alpAR < 0
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end    
    if alpMH > log(rand) || loop == 1
        h = hc;
        exph = exp(h);
        sqrtexph = sqrt(exph);
        counth = counth + 1;
    end    
        %% sample omegah2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];    
    newSh = Sh + sum(errh.^2)/2;    
    omegah2 = 1/gamrnd(newnuh, 1./newSh);
        %% sample phih
    Xphi = h(1:end-1)-muh;
    yphi = h(2:end) - muh;
    Dphi = 1/(1/Vphih + Xphi'*Xphi/omegah2);
    phihat = Dphi*(phih0/Vphih + Xphi'*yphi/omegah2);
    phic = phihat + sqrt(Dphi)*randn;
    g = @(x) -.5*log(omegah2./(1-x.^2))-.5*(1-x.^2)/omegah2*(h(1)-muh)^2;
    if abs(phic)<.9999
        alpMH = exp(g(phic)-g(phih));
        if alpMH>rand
            phih = phic;
            countphi = countphi+1;
            Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
        end
    end    
        %% sample muh
    Dmuh = 1/(1/Vmuh + ((T-1)*(1-phih)^2 + (1-phih^2))/omegah2);
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;   

    if loop>burnin
        i = loop-burnin;        
        store_h(i,:) = h';         
        store_theta(i,:) = [mu muh phih omegah2 kappa delta(1) exp(delta(2))];        
        
           % compute Q stats 
        u = (y-mu-q.*k)./exp(h/2);
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

hhat = mean(exp(store_h/2))';  %% plot std dev
thetahat = mean(store_theta)';
Qhat = mean(store_Q)';
Qstd = std(store_Q)';
thetastd = std(store_theta)';
accept = [counth countkappa countdelta countphi]/nloop;

figure;    
plot(tid, hhat, 'LineWidth',2,'Color','black'); box off;
title('exp(h_t/2)');

%% compute ml
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');    
    [ml mlstd] = ml_sv_j(y,store_theta,hhat,prior,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );    
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_h        | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('phi_h       | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('kappa       | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('mu_k        | %.2f (%.2f)\n', thetahat(6), thetastd(6)); 
fprintf('sigma2_k    | %.2f (%.2f)\n', thetahat(7), thetastd(7)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end

