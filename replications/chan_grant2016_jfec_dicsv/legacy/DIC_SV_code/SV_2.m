% This script estimates a SV model with an AR(2) log volatility process
% (SV-2).
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). On the Observed-Data Deviance
% Information Criterion for Volatility Modeling , Journal of Financial 
% Econometrics, forthcoming.

    %% prior   
phih0 = .97; Vphih = .1^2;
rhoh0 = 0; Vrhoh = 1;
mu0 = 0; Vmu = 10;
muh0 = -10; Vmuh = 10;
nuh = 5; Sh = .2^2*(nuh-1);

N = 10^5; %% estimate the prob that (phih,rhoh) is in the stationarity region
temp = [phih0+sqrt(Vphih)*randn(N,1) rhoh0+sqrt(Vrhoh)*randn(N,1)];
count = sum((sum(temp,2)< .999)&(temp(:,2)-temp(:,1)<.999)&(abs(temp(:,2))<.999));
phih_const = -log(count/N);
prior = @(m,mh,ph,rh,oh) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
    -.5*log(2*pi*Vmuh) - .5*(mh-muh0)^2/Vmuh ...
    -.5*log(2*pi*Vphih) -.5*log(2*pi*Vrhoh) + log(phih_const) ...
    -.5*(ph-phih0)^2/Vphih -.5*(rh-rhoh0)^2/Vrhoh ...
    + nuh*log(Sh) - gammaln(nuh) - (nuh+1)*log(oh) - Sh/oh;

    %% initialize the Markov chain
muh = log(var(y)); 
phih = .95; 
rhoh = 0;
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(T,1);
exph = exp(h);
    %% initialize for storage
store_theta = zeros(R*(nloop - burnin),5); % [mu muh phih rhoh omegah2]
store_h = zeros(R*(nloop - burnin),T);
store_llike = zeros(nloop-burnin,1);
store_lpost = zeros(nloop-burnin,1);
store_DIC = zeros(R,1);
store_pD = zeros(R,1);
store_ml = zeros(R,1);
    %% compute a few things outside the loop
Hthetah = speye(T) - sparse(3:T,2:(T-1),phih*ones(1,T-2),T,T) ...
    - sparse(3:T,1:(T-2),rhoh*ones(1,T-2),T,T);
newnuh = T/2 + nuh;
counth = 0; countthetah = 0;

randn('seed',sum(clock*100)); rand('seed',sum(clock*1000)); 
disp('Starting SV-2.... ');
disp(' ' );

start_time = clock;

for bigloop = 1:R
    disp(  [ num2str( R-bigloop+1) ' big loops to go... ' ] )
for loop = 1:nloop 
        %% sample mu    
    invDmu = 1/Vmu + sum(1./exph);
    muhat = invDmu\sum(y./exph);   
    mu = muhat + chol(invDmu,'lower')'\randn;           
        %% sample h    
    intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));    
    HiSH = Hthetah'*sparse(1:T,1:T,[1/intvar;1/intvar;1/omegah2*ones(T-2,1)])*Hthetah;
    deltah = Hthetah\[muh;muh;muh*(1-phih-rhoh)*ones(T-2,1)];
    HiSHdeltah = HiSH*deltah;
    s2 = (y-mu).^2;
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
    hstar = ht;    
    lph = @(x) -.5*(x-deltah)'*HiSH*(x-deltah) -.5*sum(x) -.5*exp(-x)'*s2;
    logc = lph(ht) + log(4);        
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
    temp1 = (1-rhoh)/((1+rhoh)*((1-rhoh)^2-phih^2));
    errh = [(h(1)-muh)/sqrt(temp1); (h(2)-muh)/sqrt(temp1); ...
        h(3:end)-phih*h(2:end-1)-rhoh*h(1:end-2)-muh*(1-phih-rhoh)];    
    newSh = Sh + sum(errh.^2)/2;    
    omegah2 = 1/gamrnd(newnuh, 1./newSh);    
    
    %% sample phih and rhoh
    Xthetah = [h(2:end-1)-muh h(1:end-2)-muh];
    ythetah = h(3:end) - muh;
    invDthetah = diag([1/Vphih 1/Vrhoh]) + Xthetah'*Xthetah/omegah2;
    thetahhat = invDthetah\([phih0/Vphih;rhoh0/Vrhoh] + Xthetah'*ythetah/omegah2);
    thetahc = thetahhat + chol(invDthetah,'lower')\randn(2,1);        
    g = @(x) -log((1-x(2))*omegah2/((1+x(2))*((1-x(2))^2-x(1)^2))) ...
        -.5/((1-x(2))*omegah2/((1+x(2))*((1-x(2))^2-x(1)^2)))*(h(1)+h(2)-2*muh)^2;    
    if sum(thetahc)< .999 && thetahc(2)-thetahc(1)<.999 && abs(thetahc(2))<.999
        alpMH = exp(g(thetahc)-g([phih rhoh]));
        if alpMH>rand
            phih = thetahc(1);
            rhoh = thetahc(2);
            countthetah = countthetah+1;
            Hthetah = speye(T) - sparse(3:T,2:(T-1),phih*ones(1,T-2),T,T) ...
                - sparse(3:T,1:(T-2),rhoh*ones(1,T-2),T,T);
        end
    end
    
    %% sample muh
    intvar = (1-rhoh)*omegah2/((1+rhoh)*((1-rhoh)^2-phih^2));
    Dmuh = 1/(1/Vmuh + (T-2)*(1-phih-rhoh)^2/omegah2 + 2/intvar);
    muhhat = Dmuh*(muh0/Vmuh + (h(1)+h(2))/intvar ...
        + (1-phih-rhoh)/omegah2*sum(h(3:end)-phih*h(2:end-1)-rhoh*h(1:end-2)));
    muh = muhhat + sqrt(Dmuh)*randn;   

    if loop>burnin
        i = loop-burnin;        
        store_h((bigloop-1)*(nloop-burnin)+i,:) = h';         
        store_theta((bigloop-1)*(nloop-burnin)+i,:) =  [mu muh phih rhoh omegah2];
        s2 = (y-mu).^2;
        if  DIC_choice == 1 % observed-data DIC
            llike = intlike_sv_2(s2,muh,phih,rhoh,omegah2,ht,50);
            lpost = llike + prior(mu,muh,phih,rhoh,omegah2);
        elseif DIC_choice == 2; % conditional DIC
            llike = conlike_sv_2(s2,h);
            lpost = like_sv_2(s2,h,muh,phih,rhoh,omegah2) + prior(mu,muh,phih,rhoh,omegah2);
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
accept = [counth/nloop countthetah/nloop];

figure;    
plot(tid, hhat, 'LineWidth',2,'Color','black'); box off;
title('exp(h_t/2)');

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_h        | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 
fprintf('phi_h       | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('rho_h       | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 

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


