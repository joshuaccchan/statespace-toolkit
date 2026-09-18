% This script estimates the SV-t model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

    %% prior
nuub = 100; %% upperbound for nu
phih0 = .97; Vphih = .1^2;
mu0 = 0; Vmu = 10;
muh0 = -10; Vmuh = 10;
nuh = 5; Sh = .2^2*(nuh-1);
phih_const = 1/(normcdf(1,phih0,sqrt(Vphih))-normcdf(-1,phih0,sqrt(Vphih)));
prior = @(m,mh,ph,oh,n) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
     -.5*log(2*pi*Vmuh) - .5*(mh-muh0)^2/Vmuh ... 
     -.5*log(2*pi*Vphih) + log(phih_const) -.5*(ph-phih0)^2/Vphih ...
     + nuh*log(Sh) - gammaln(nuh) - (nuh+1)*log(oh) - Sh/oh + log(1/(nuub-2));
    %% initialize the Markov chain
nu = 5;
lam = 1./gamrnd(nu/2,2/nu,T,1);
muh = log(var(y)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(T,1);
exph = exp(h);
    %% initialize for storage
store_theta = zeros(R*(nloop - burnin),5); % [mu muh phih omegah2 nu]
store_h = zeros(R*(nloop - burnin),T);
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);
store_pD = zeros(R,1);
store_ml = zeros(R,1);
    %% compute a few things outside the loop
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
newnuh = T/2 + nuh;
counth = 0; countphi = 0; countnu = 0;
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000)); 

disp('Starting SV-t.... ');
disp(' ' );
start_time = clock;
for bigloop = 1:R
    disp(  [ num2str( R-bigloop+1) ' big loops to go... ' ] )
for loop = 1:nloop 
        %% sample mu    
    invDmu = 1/Vmu + sum(1./(exph.*lam));
    muhat = invDmu\sum(y./(exph.*lam));   
    mu = muhat + chol(invDmu,'lower')'\randn; 
        %% sample h    
    HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
    HiSHdeltah = HiSH*deltah;
    s2 = (y-mu).^2./lam;
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
    logc = lph(ht) + log(5);     
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
        %% sample lam
    lam = 1./gamrnd((nu+1)/2,2./(nu+(y-mu).^2./exph));
        %% sample nu                
    sum1 = sum(log(lam));
    sum2 = sum(1./lam);
    fnu = @(x) T*(x/2.*log(x/2)-gammaln(x/2)) - (x/2+1)*sum1 - x/2*sum2;
    dfnu = @(x) T/2*(log(x/2)+1-psi(x/2)) - .5*(sum1+sum2);  % score
    d2fnu = @(x) T/(2*x) - T/4*psi(1,x/2);                   % Hessian
    err_nu = 1;
    nut = nu;
    while abs(err_nu) > 10^(-3) % stopping criteria
        err_nu = dfnu(nut);
        Knu = -d2fnu(nut);      % infomation matrix
        nut = nut + Knu\err_nu;
        if nut<2
            nut = 5;
            Knu = -f2(nut);
            break;
        end
    end
    sqrtDnu = sqrt(1/Knu);
    nuc = nut + sqrtDnu*randn; 
    if nuc>2 && nuc < nuub
        alpMH = exp(fnu(nuc)-fnu(nu)) ... 
            * normpdf(nu,nut,sqrtDnu)/normpdf(nuc,nut,sqrtDnu);
        if alpMH > rand
            nu = nuc;
            countnu = countnu+1;
        end    
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
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + ...
        (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn; 
    
    if loop>burnin
        i = loop-burnin;        
        store_h((bigloop-1)*(nloop-burnin)+i,:) = h';         
        store_theta((bigloop-1)*(nloop-burnin)+i,:) =  [mu muh phih omegah2 nu];        
        s2 = (y-mu).^2;        
        if  DIC_choice == 1 % observed-data DIC
            llike = intlike_sv_t(s2,muh,phih,omegah2,nu,ht,50);
            lpost = llike + prior(mu,muh,phih,omegah2,nu);
        elseif DIC_choice == 2; % conditional DIC
            llike = conlike_sv_t(s2,h,lam);
            lpost = like_sv_t(s2,h,lam,muh,phih,omegah2,nu) + prior(mu,muh,phih,omegah2,nu);
        end 
        if DIC_choice == 1 || DIC_choice == 2
            store_llike(i,:) = llike; 
            store_lpost(i,:) = lpost;            
        end
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
accept = [counth/nloop countphi/nloop countnu/nloop];

figure;    
plot(tid, hhat, 'LineWidth',2,'Color','black'); box off;
title('exp(h_t/2)');

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_h        | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('phi_h       | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('nu          | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 


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


