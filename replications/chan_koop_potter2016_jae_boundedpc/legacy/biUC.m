% % =======================================================================
% % Bivariate Unobserved components model with stochastic volatility 
% %
% % y_t - taupi_t = rhopi_t(y_t-1 - taupi_t-1) + lam_t(u_t-tauu_t) + epi_t,
% % u_t - tauu_t = rhou1_(u_t-1 - tauu_t-1) + rhou2_(u_t-2 - tauu_t-2) + eu_t
% % epi_t ~ N(0,exp(h_t))
% % 0 < taupi_t < 5
% % 4 < tauu_t  < 7
% % -1< lam_t   < 0
% % 0 < rhopi_t < 1
% % 
% % See Chan, J.C.C., Koop, G. and Potter, S.M. (2016). A Bounded Model of 
% % Time Variation in Trend Inflation, NAIRU and the Phillips Curve, 
% % Journal of Applied Econometrics, 31(3), 551-565.
% % 
% % (c) Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
bound_taupi = 1;  % 1: impose bounds on inflation trend; 0: no bounds
bound_tauu = 1;   % 1: impose bounds on NAIRU; 0: no bounds
tvp_lam = 1;      % 1: lambda is time-varying; 0: lambda is approximately constant
sv = 1;           % 1: with SV; 0: variance in the inflation eq is approximately constant
nsim = 40000;
burnin = 1000;
load 'USdata.csv';
y0 = USdata(2,1);
y = USdata(3:end,1); 
u0 = USdata(1:2,2);
u = USdata(3:end,2);
T = length(y);

% priors
if bound_taupi == 1
    api = 0; bpi = 5;        % bounds for taupi
elseif bound_taupi == 0
    api = -10^6; bpi = 10^6; 
end
if bound_tauu == 1
    au = 4; bu = 7;          % bounds for tauu
elseif bound_tauu == 0
    au = -10^6; bu = 10^6;    
end    
taupi0 = 3; invVtaupi = 1/5; sqrtinvVtaupi = sqrt(invVtaupi);
tauu0 = [5; 5]; invVtauu = 1/5; sqrtinvVtauu = sqrt(invVtauu);
invVrhopi = 1; sqrtinvVrhopi = sqrt(invVrhopi);
rhou0 = [1.8 -.8]';
invVrhou = speye(2)/5;
invVlam = 1; sqrtinvVlam = sqrt(invVlam);
invVh = 1/5; 
nutaupi0 = 10; Staupi0 = .02*(nutaupi0-1);
nutauu0 = 10; Stauu0 = .01*(nutauu0-1);
nurhopi0 = 10; Srhopi0 = .002*(nurhopi0-1);
nulam0 = 10; Slam0 = .002*(nulam0-1);
nuh0 = 10; Sh0 = .1*(nuh0-1);
nuu0 = 10; Su0 = .5*(nuu0-1);

% compute and define a few things 
id1 = (1:T)';
id2 = (1:T-1)';
Kh =  speye(T) - spdiags(ones(T-1,1),-1,T,T);
newnuu = (T-2)/2 + nuu0;
newnuh = (T-1)/2 + nuh0;
newnutaupi = (T-1)/2 + nutaupi0;
newnutauu = (T-1)/2 + nutauu0;
newnurhopi = (T-1)/2 + nurhopi0;
newnulam = (T-1)/2 + nulam0;
 
% initialize for storage 
store_sig2 = zeros(nsim,6); % [u,h,taupi,tauu,rhopi,lam]
store_taupi = zeros(nsim,T); 
store_tauu = zeros(nsim,T); 
store_rhopi = zeros(nsim,T);
store_rhou = zeros(nsim,2);
store_lam = zeros(nsim,T);
store_h = zeros(nsim,T);
countstate = zeros(5,1);  % [taupi,tauu,rhopi,lam,h]
countsig2 = zeros(4,1);   % [taupi,tauu,rhopi,lam]

% initialize the Markov chain 
initialize_MC;

% MCMC starts here 
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));

disp('Model specification: ');
if bound_taupi == 1
   disp(['  - inflation trend is bounded between (' num2str(api) ',' num2str(bpi) ')' ] );
elseif bound_taupi == 0
    disp('  - no bounds on inflation trend');
end
if bound_tauu == 1
   disp(['  - NAIRU is bounded between (' num2str(au) ',' num2str(bu) ')' ] );
elseif bound_tauu == 0
    disp('  - no bounds on NAIRU');
end
if tvp_lam == 1
    disp('  - lambda is time-varying');
elseif tvp_lam == 0
     disp('  - lambda is approximately constant');
end
if sv == 1
    disp('  - with stochastic volatility');
elseif sv == 0
     disp('  - without stochastic volatility');
end

disp(' ');    
disp('Starting MCMC.... ');

start_time = clock;

for isim = 1:nsim + burnin
  
        % sample taupi    
    Kpi = speye(T) - sparse(2:T,1:(T-1),rhopi(2:T),T,T); 
    mupi = Kpi\([rhopi(1)*(y0-taupi0); sparse(T-1,1)] + lam.*(u-tauu)) ;
    expinvh = exp(-h);    
    alppi = Kh\[taupi0; sparse(T-1,1)];
    Staupi = Kh'*sparse(id1,id1,[invVtaupi repmat(1/sigtaupi2,1,T-1)],T,T)*Kh;
    Sy = Kpi'*sparse(id1,id1,expinvh,T,T)*Kpi;
    Hpi =  Sy + Staupi;
    taut = Hpi\(Staupi*alppi + Sy*(y-mupi));
    taut = min(taut,bpi-.05);    taut = max(taut,api+.05); 
    cholHpi = chol(Hpi,'lower'); 
        % AR step
            % compute the constant c
    taustar = taut;    
    logc = -sum(log(normcdf((bpi-taustar(1:end-1))/sigtaupi)-normcdf((api-taustar(1:end-1))/sigtaupi))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        tauc = taut + cholHpi'\randn(T,1);
        if  max(tauc) <= bpi - .02 && min(tauc) >= api + .02
            alpARc = -sum(log(normcdf((bpi-tauc(1:end-1))/sigtaupi)-normcdf((api-tauc(1:end-1))/sigtaupi))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end      
    if flag == 1
        alpAR = - sum(log(normcdf((bpi-taupi(1:end-1))/sigtaupi)-normcdf((api-taupi(1:end-1))/sigtaupi))) - logc;
        if isim == 1
            alpMH=1;
        elseif alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end        
        if alpMH > log(rand)
            taupi = tauc;
            countstate(1) = countstate(1) + 1;
        end     
    end
    
        % sample tauu
    Lam = sparse(id1,id1,-lam);
    pistar = y-taupi;
    z = pistar - rhopi.*[(y0-taupi0);pistar(id2)] - lam.*u;
    tempSpi = Lam'*sparse(id1,id1,expinvh);
    Spi = tempSpi*Lam;
    Ku = spdiags([ones(T,1) [-rhou(1)*ones(T-1,1); 0] [-rhou(2)*ones(T-2,1);0;0]], [0 -1 -2], T, T);   
    muu = Ku\[rhou(1)*(u0(2)-tauu0(2))+rhou(2)*(u0(1)-tauu0(1)); rhou(2)*(u0(2)-tauu0(2)); sparse(T-2,1)];    
    alpu = Kh\[tauu0(2); sparse(T-1,1)];
    Su  = Ku'*sparse(id1,id1,[1/5 1/5 repmat(1/sigu2,1,T-2)])*Ku;     %% u1 and u2 have variances 5
    Stauu = Kh'*sparse(id1,id1,[invVtauu repmat(1/sigtauu2,1,T-1)])*Kh;
    Hu =  Su + Stauu + Spi;
    taut = Hu\(Stauu*alpu + Su*(u-muu) + tempSpi*z);
    taut = min(taut,bu-.08);    taut = max(taut,au+.05); 
    cholHu = chol(Hu,'lower'); 
        % AR step
            % compute the constant c
    taustar = taut;
    logc = -sum(log(normcdf((bu-taustar(1:end-1))/sigtauu)-normcdf((au-taustar(1:end-1))/sigtauu))) + log(3);
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        tauc = taut + cholHu'\randn(T,1);
        if  max(tauc) <= bu - .02 && min(tauc) >= au + .02
            alpARc =  -sum(log(normcdf((bu-tauc(1:end-1))/sigtauu)-normcdf((au-tauc(1:end-1))/sigtauu))) - logc;
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end      
    if flag == 1     
        alpAR = - sum(log(normcdf((bu-tauu(1:end-1))/sigtauu)-normcdf((au-tauu(1:end-1))/sigtauu))) - logc;
        if isim == 1 
            alpMH = 1;
        elseif alpAR < 0
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end        
        if alpMH > log(rand)
            tauu = tauc;
            countstate(2) = countstate(2) + 1;
        end     
    end 
    
        % sample rhopi
    pistar = y - taupi;
    ustar = u - tauu;
    XinvSy = [y0-taupi0; pistar(id2)].*expinvh;
    XinvSyX = XinvSy.*[(y0-taupi0); pistar(id2)]; 
    Hrhopi = sparse(id1,id1,XinvSyX) + Kh'*sparse(id1,id1,[invVrhopi repmat(1/sigrhopi2,1,T-1)])*Kh;
    rhot = Hrhopi\(XinvSy.*(pistar+Lam*ustar));    
    rhot = min(rhot,1-.05);    rhot = max(rhot,.02);
    cholHrhopi = chol(Hrhopi,'lower'); 
        % AR step
            % compute the constant c
    sigrhopi = sqrt(sigrhopi2);
    rhostar = rhot;
    logc = -sum(log(normcdf((1-rhostar(1:end-1))/sigrhopi)-normcdf(-rhostar(1:end-1)/sigrhopi))) + log(3); 
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        rhoc = rhot + cholHrhopi'\randn(T,1);
        if  max(rhoc)<= 1 -.005 && min(rhoc)>= .01
            alpARc = -sum(log(normcdf((1-rhoc(1:end-1))/sigrhopi)-normcdf(-rhoc(1:end-1)/sigrhopi))) - logc; 
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end      
    if flag == 1
        alpAR = -sum(log(normcdf((1-rhopi(1:end-1))/sigrhopi)-normcdf(-rhopi(1:end-1)/sigrhopi))) - logc;
        if isim == 1
            alpMH = 1;
        elseif alpAR < 0 
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end        
        if alpMH > log(rand)
            rhopi = rhoc;
            countstate(3) = countstate(3) + 1;
        end
    end    
    
        % sample rhou
    Xu = [[u0(2)-tauu0(2); ustar(id2)] [u0-tauu0; ustar(1:T-2)]];  
    XuinvSu = Xu'*sparse(1:T,1:T,[1/5 1/5 1/sigu2*ones(1,(T-2))]);
    invDrhou = invVrhou + XuinvSu*Xu;
    drhou = invVrhou*rhou0 + XuinvSu*ustar;
    rhouhat = invDrhou\drhou;
    rhou = rhouhat + chol(invDrhou,'lower')'\randn(2,1);
    
        % sample lam    
    w = pistar - rhopi.*[y0-taupi0; pistar(id2)];
    XinvSy = ustar.*expinvh;
    XinvSyX = XinvSy.*ustar;
    Hlam = sparse(id1,id1,XinvSyX) + Kh'*sparse(id1,id1,[invVlam repmat(1/siglam2,1,T-1)])*Kh;
    lamt = Hlam\(XinvSy.*w);    
    lamt = max(lamt,-1+.05);    lamt = min(lamt,-.05);
    cholHlam = chol(Hlam,'lower'); 
        % AR step
            % compute the constant c
    siglam = sqrt(siglam2);
    lamstar = lamt;
    logc = -sum(log(normcdf(-lamstar(1:end-1)/siglam)-normcdf((-1-lamstar(1:end-1))/siglam))) + log(3); 
    flag = 0; count = 0;
    while flag == 0 && count < 1000 % give up when count >= 1000
        lamc = lamt + cholHlam'\randn(T,1);
        if  max(lamc)<= -.01 && min(lamc)>= -1+.01
            alpARc = -sum(log(normcdf((-lamc(1:end-1))/siglam)-normcdf((-1-lamc(1:end-1))/siglam))) - logc; 
            if alpARc > log(rand)
                flag = 1;
            end
        end
        count = count + 1;
    end      
    if flag == 1
        alpAR = -sum(log(normcdf(-lam(1:end-1)/siglam)-normcdf((-1-lam(1:end-1))/siglam))) - logc;
        if alpAR < 0 
            alpMH = 1;
        elseif alpARc < 0
            alpMH = - alpAR;
        else
            alpMH = alpARc - alpAR;
        end        
        if alpMH > log(rand)
            lam = lamc;
            countstate(4) = countstate(4) + 1;
        end
    end    
    
        % sample h    
    Sh = spdiags([invVh; 1/sigh2*ones(T-1,1)],0,T,T);
    KSKh = Kh'*Sh*Kh;    
    s2 = (w-lam.*ustar).^2;
    errh = 1; ht = h;
    while errh> 10^(-3)
        expht = exp(ht);
        sexpht = s2./expht;
        fh = -.5 + .5*sexpht;
        Gh = .5*sexpht;
        Hh = KSKh + spdiags(Gh,0,T,T);
        newht = Hh\(fh+Gh.*ht);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Hh,'lower');
    % AR-step:     
    hstar = ht;
    logc = -.5*hstar'*KSKh*hstar -.5*sum(hstar) - .5*exp(-hstar)'*s2 + log(3);
    flag = 0;
    while flag == 0
        hc = ht + cholHh'\randn(T,1);
        vhc = hc-ht;        
        alpARc = -.5*hc'*KSKh*hc -.5*sum(hc) -.5*exp(-hc)'*s2 + ...
            + .5*vhc'*Hh*vhc - logc;            
        if alpARc > log(rand)
            flag = 1;
        end
    end        
    % MH-step
    vh = h-ht;
    alpAR = -.5*h'*KSKh*h -.5*sum(h) -.5*exp(-h)'*s2 +...
        .5*vh'*Hh*vh - logc;
    if alpAR < 0
        alpMH = 1;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end    
    if alpMH > log(rand) || isim == 1
        h = hc;
        countstate(5) = countstate(5) + 1;
    end
    
        % sample sigtaupi2    
    errsigtaupi = taupi(2:end) - taupi(1:end-1);
    newStaupi = Staupi0 + sum(errsigtaupi.^2)/2;
    sigtaupi2c = 1/gamrnd(newnutaupi, 1./newStaupi);
    sigtaupic = sqrt(sigtaupi2c);
    sigtaupi = sqrt(sigtaupi2);
    sigapi = api - taupi(1:end-1); 
    sigbpi = bpi - taupi(1:end-1);    
    alpMH = -sum(log(normcdf(sigbpi/sigtaupic)-normcdf(sigapi/sigtaupic))) + ...
        sum(log(normcdf(sigbpi/sigtaupi)-normcdf(sigapi/sigtaupi)));
    if alpMH > log(rand)
        sigtaupi2 = sigtaupi2c;         
        sigtaupi = sigtaupic;
        countsig2(1) = countsig2(1) + 1;
    end
    
        % sample sigtauu2
    errsigtauu = tauu(2:end) - tauu(1:end-1);
    newStauu = Stauu0 + sum(errsigtauu.^2)/2;
    sigtauu2c = 1/gamrnd(newnutauu, 1./newStauu);
    sigtauuc = sqrt(sigtauu2c);
    sigtauu = sqrt(sigtauu2);
    sigau = au - tauu(1:end-1); 
    sigbu = bu - tauu(1:end-1);
    
    alpMH = -sum(log(normcdf(sigbu/sigtauuc)-normcdf(sigau/sigtauuc))) + ...
        sum(log(normcdf(sigbu/sigtauu)-normcdf(sigau/sigtauu)));
    if alpMH > log(rand)
        sigtauu2 = sigtauu2c;         
        sigtauu = sigtauuc;
        countsig2(2) = countsig2(2) + 1;
    end
    
        % sample sigrhopi2
    errsigrhopi = rhopi(2:end) - rhopi(1:end-1);
    newSrhopi = Srhopi0 + sum(errsigrhopi.^2)/2;
    sigrhopi2c = 1/gamrnd(newnurhopi, 1./newSrhopi);
    sigrhopic = sqrt(sigrhopi2c);
    sigrhopi = sqrt(sigrhopi2);
    sigapi = -rhopi(1:end-1); 
    sigbpi = 1-rhopi(1:end-1);    
    alpMH = -sum(log(normcdf(sigbpi/sigrhopic)-normcdf(sigapi/sigrhopic))) + ...
        sum(log(normcdf(sigbpi/sigrhopi)-normcdf(sigapi/sigrhopi)));
    if alpMH > log(rand)
        sigrhopi2 = sigrhopi2c;         
        sigrhopi = sigrhopic;
        countsig2(3) = countsig2(3) + 1;
    end
    
        % sample siglam2
    if tvp_lam == 1    
        errsiglam = lam(2:end) - lam(1:end-1);
        newSlam = Slam0 + sum(errsiglam.^2)/2;
        siglam2c = 1/gamrnd(newnulam, 1./newSlam);
        siglamc = sqrt(siglam2c);
        siglam = sqrt(siglam2);
        sigau = -1-lam(1:end-1);
        sigbu = -lam(1:end-1);
        alpMH = -sum(log(normcdf(sigbu/siglamc)-normcdf(sigau/siglamc))) + ...
            sum(log(normcdf(sigbu/siglam)-normcdf(sigau/siglam)));
        if alpMH > log(rand)
            siglam2 = siglam2c;
            siglam = siglamc;
            countsig2(4) = countsig2(4) + 1;
        end
    end  
    
        % sample sigh2
    if sv == 1
        errh = h(2:end) - h(1:end-1);
        newS = Sh0 + sum(errh.^2)/2;
        sigh2 = 1/gamrnd(newnuh, 1./newS);
    end

        % sample sigu2
    erru = ustar - Xu*rhou;
    newS = Su0 + sum(erru(3:end).^2)/2;
    sigu2 = 1/gamrnd(newnuu, 1./newS);    

    if isim>burnin
        isave = isim-burnin;
        store_taupi(isave,:) = taupi';
        store_tauu(isave,:) = tauu';
        store_rhopi(isave,:) = rhopi';
        store_rhou(isave,:) = rhou';
        store_lam(isave,:) = lam';
        store_h(isave,:) = h'; 
        store_sig2(isave,:) = [sigu2 sigh2 sigtaupi2 sigtauu2 sigrhopi2 siglam2];
    end    
    if (mod(isim, 2000) == 0)
        disp([num2str(isim) ' loops... '])
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

accept_rate = countstate/(nsim+burnin);

% plots of various states 
taupihat = mean(store_taupi)';
taupilb = quantile(store_taupi,.05)';
taupiub = quantile(store_taupi,.95)';
tauuhat = mean(store_tauu)';
tauulb = quantile(store_tauu,.05)';
tauuub = quantile(store_tauu,.95)';
rhopihat = mean(store_rhopi)';
rhopilb = quantile(store_rhopi,.05)';
rhopiub = quantile(store_rhopi,.95)';
rhouhat = mean(store_rhou)';
lamhat = mean(store_lam)';
lamlb = quantile(store_lam,.05)';
lamub = quantile(store_lam,.95)';
hhat = mean(store_h)'; 
hlb = quantile(store_h,.05)';
hub = quantile(store_h,.95)';

tid = (1948.5:.25:2013)';   
figure;
subplot(1,2,1); 
hold on        
    plot(tid, taupihat, 'LineWidth',1,'Color','blue');
    plot(tid, [taupilb taupiub],':', 'LineWidth',1 ,'Color','red'); 
hold off 
xlim([1947 2014]);  title('\tau^\pi');  box off;
subplot(1,2,2); 
hold on
    plot(tid, tauuhat, 'LineWidth',1,'Color','blue');
    plot(tid, [tauulb tauuub],':', 'LineWidth',1 ,'Color','red');        
hold off
xlim([1947 2014]); title('\tau^u');  box off;
set(gcf,'Position',[100 100 800 300]);   
 
figure;
subplot(1,2,1); 
hold on
    plot(tid, rhopihat, 'LineWidth',1,'Color','blue');
    plot(tid, [rhopilb rhopiub],':', 'LineWidth',1 ,'Color','red');        
hold off
xlim([1947 2014]); title('\rho^\pi');  box off;
subplot(1,2,2); 
hold on        
    plot(tid, lamhat, 'LineWidth',1,'Color','blue');
    plot(tid, [lamlb lamub],':', 'LineWidth',1 ,'Color','red');        
hold off
xlim([1947 2014]); title('\lambda');  box off;
set(gcf,'Position',[100 100 800 300]);   
 
figure 
hold on    
    plot(tid, hhat, 'LineWidth',1,'Color','blue');
    plot(tid, [hlb hub],':', 'LineWidth',1 ,'Color','red');        
hold off
xlim([1947 2014]); title('h');  box off;