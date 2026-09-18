% This script estimates the SV-M model.
% See:
% 
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

    %% prior
phih0 = .97; Vphih = .1^2;
mu0 = 0; Vmu = 10;
muh0 = -10; Vmuh = 10;
alp0 = 0; Valp = 100^2; 
nuh = 5; Sh = .2^2*(nuh-1);
phih_const = 1/(normcdf(1,phih0,sqrt(Vphih))-normcdf(-1,phih0,sqrt(Vphih)));
prior = @(m,a,mh,ph,oh) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
    -.5*log(2*pi*Valp) -.5*(a-alp0)^2/Valp ...
    -.5*log(2*pi*Vmuh) -.5*(mh-muh0)^2/Vmuh ...
    -.5*log(2*pi*Vphih) + log(phih_const) -.5*(ph-phih0)^2/Vphih ...
    + nuh*log(Sh) - gammaln(nuh) - (nuh+1)*log(oh) - Sh/oh;

    %% initialize the Markov chain
alp = 0;
mu = mean(y);
muh = log(var(y)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(T,1);
exph = exp(h);
    %% initialize for storeage
store_theta = zeros(R*(nloop - burnin),5); % [mu alp muh phih omegah2]
store_h = zeros(R*(nloop - burnin),T);
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);
store_ml = zeros(R,1);
store_pD = zeros(R,1);
    %% compute a few things outside the loop
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);    
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
newnuh = T/2 + nuh;
counth = 0; countphih = 0;
disp('Starting SV-M.... ');
disp(' ' );

start_time = clock;

for bigloop = 1:R
    disp(  [ num2str( R-bigloop+1) ' big loops to go... ' ] )
for loop = 1:nloop 
        %% sample mu and alp
    X = [ones(T,1) exph];
    invSy = sparse(1:T,1:T,1./exph);
    XinvSy = X'*invSy;
    invDbeta = sparse(1:2,1:2,[1/Vmu 1/Valp]) + XinvSy*X;
    betahat = invDbeta\([mu0/Vmu; alp0/Valp] + XinvSy*y);
    beta = betahat + chol(invDbeta,'lower')'\randn(2,1);
    mu = beta(1); 
    alp = beta(2);
        %% sample h    
    HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
    HinvSHdeltah = HiSH*deltah;
    s2 = (y-mu).^2;
    errh = 1; ht = h;
    while errh> 10^(-3);
        expht = exp(ht);
        sinvexpht = s2./expht;
        lam2expht = alp^2.*expht;
        fh = -.5 + .5*sinvexpht - .5*lam2expht;
        Gh = .5*sinvexpht + .5*lam2expht;
        Kh = HiSH + sparse(1:T,1:T,Gh);
        newht = Kh\(fh+Gh.*ht+HinvSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Kh,'lower');
    % AR-step:     
    hstar = ht;
    lph = @(x) -.5*(x-deltah)'*HiSH*(x-deltah) -.5*sum(x) ...
        -.5*exp(-x)'*(y-mu-alp*exp(x)).^2;
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
            countphih = countphih + 1;
            Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
        end
    end    
        %% sample muh    
    Dmuh = 1/(1/Vmuh + ((T-1)*(1-phih)^2 + (1-phih^2))/omegah2);
    muhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhat + sqrt(Dmuh)*randn;   
    if loop>burnin
        i = loop-burnin;     
        store_h((bigloop-1)*(nloop-burnin)+i,:)  = h';         
        store_theta((bigloop-1)*(nloop-burnin)+i,:) = [mu alp muh phih omegah2];
        s2 = (y-mu-alp*exp(h)).^2;
        if  DIC_choice == 1 % observed-data DIC
            llike = intlike_sv_m(y,mu,alp,muh,phih,omegah2,ht,50);
            lpost = llike + prior(mu,alp,muh,phih,omegah2);  
        elseif DIC_choice == 2; % conditional DIC        
            llike = conlike_sv_m(s2,h);
            lpost = like_sv_m(s2,h,muh,phih,omegah2) + prior(mu,alp,muh,phih,omegah2);
        end
        store_llike(i,:) = llike; 
        store_lpost(i,:) = lpost;    
    end    
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
end
%% compute DIC
[~,id] = max(store_lpost);
DIC = -4*mean(store_llike) + 2*store_llike(id);
pD = -2*mean(store_llike)  + 2*store_llike(id);
store_DIC(bigloop) = DIC;
store_pD(bigloop) = pD;

end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

hhat = mean(exp(store_h/2))';  %% plot std dev
thetahat = mean(store_theta)';
thetastd = std(store_theta)';
accept = [counth/nloop countphih/nloop];

figure;    
plot(tid, hhat, 'LineWidth',2,'Color','black'); box off;
title('exp(h_t/2)');

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_h        | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('phi_h       | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('alpha       | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 

DIC = mean(store_DIC);
DICstd = std(store_DIC)/sqrt(R);
pD = mean(store_pD);
pDstd = std(store_pD)/sqrt(R);

fprintf('\n'); 
if DIC_choice == 1   % 1: ; 2: conditional DIC
    fprintf('observed-data DIC: %.1f (%.2f)\n', DIC, DICstd); 
    fprintf('effective # of parameters: %.1f (%.2f)\n', pD, pDstd); 
elseif DIC_choice == 2
    fprintf('conditional DIC: %.1f (%.2f)\n', DIC, DICstd); 
    fprintf('effective # of parameters: %.1f (%.2f)\n', pD, pDstd); 
end







