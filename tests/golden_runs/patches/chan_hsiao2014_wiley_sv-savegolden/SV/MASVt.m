%% GOLDEN-RUN VARIANT PATCH (2026-09-19): identical to legacy/SV/MASVt.m except that
%% a numeric capture is appended at the end, which prints the results the script computes
%% and saves their posterior means and 5% and 95% quantiles to golden_capture.mat.
%% The computation is untouched.
% % =======================================================================
% % Stochastic volatility model with  MA(1) Student-t errors
% %
% % y_t = mu + u_t,
% % u_t = epsilon_t + psi epsilon_{t-1},     epsilon_t ~ N(0,exp(h_t)*lam_t),
% % h_t = muh + phih(h_{t-1}-muh) + zeta_t,  zeta_t ~ N(0,sigh2),
% % lam_t ~ inverse-gamma(nu/2,nu/2),
% % 
% % See Chan, J.C.C. and Hsiao, C.Y.L (2014). Estimation of Stochastic
% % Volatility Models with Heavy Tails and Serial Dependence. 
% % In: I. Jeliazkov and X.S. Yang (Eds.), Bayesian Inference in the 
% % Social Sciences, 159-180, John Wiley & Sons, New York.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
nloop = 2100;
burnin = 1000;
load 'silver.csv';
y = silver; 
T = length(y);

%% prior
invVmu = 1/5;
phih0 = .95; invVphih = 1;
muh0 = 0; invVmuh = 1/5;
invVpsi = 1; 
nuh = 10; Sh = .02*(nuh-1);
nuub = 50;  % upper bound for nu

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigh2 = .05;
phih = .95;
muh = 1;
psi_p = 0;
nu = 5;
lam = 1./gamrnd(nu/2,2/nu,T,1);
mu = mean(y);
h = log(var(y)*.8)*ones(T,1);
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi_p*ones(1,T-1),T,T); 
f = @(x) fMA1(x,y-mu,h);
psihat = fminbnd(@(x) -f(x),-.99,.99);
psi_p = psihat;

% initialize for storage
store_theta = zeros(nloop - burnin,6); % store [psi_p nu mu muh phih sigh2]
store_exph = zeros(nloop - burnin,T);   % store exp(h_t/2)

%% compute a few things outside the loop
psipri = @(x) log(normpdf(x,0,sqrt(1/invVpsi)) ...
    /(normcdf(sqrt(invVpsi))-normcdf(-sqrt(invVpsi))));
nugrid = linspace(.1,nuub,201)';
countnu = 0;
countpsi = 0;

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop
        %% sample mu    
    Sigy = Hpsi*sparse(1:T,1:T,exp(h).*lam)*Hpsi';
    Dmu = 1/(invVmu + ones(1,T)*(Sigy\ones(T,1)));
    muhat = Dmu*(ones(1,T)*(Sigy\y));
    mu = muhat + sqrt(Dmu)*randn;    
        %% sample h
    Ystar = log(((Hpsi\(y-mu))./sqrt(lam)).^2 + .0001);
    [h muh phih sigh2] = SV(Ystar,h,muh,phih,sigh2,[muh0 invVmuh ...
        phih0 invVphih nuh Sh]);     
        %% sample psi_p
    f = @(x) fMA1(x,y-mu,h+log(lam)) + psipri(x);
    psihat = fminsearch(@(x) -f(x),psihat); % find the mode
    sqVpsic = .05; Vpsic = sqVpsic^2;
    psic = psihat + sqVpsic*randn;
    if abs(psic)<.9999
        alpMH = f(psic) - f(psi_p) + ...
            -.5*(psi_p-psihat)^2/Vpsic + .5*(psic-psihat)^2/Vpsic;
    else
        alpMH = -inf;
    end
    if alpMH>log(rand)
        psi_p = psic;
        Hpsi = speye(T) + sparse(2:T,1:(T-1),psi_p*ones(1,T-1),T,T); 
        countpsi = countpsi + 1;
    end    
        %% sample lam
    temp1 = (Hpsi\(y-mu)).^2./exp(h)/2;  
    lam = 1./gamrnd((nu+1)/2,1./(nu/2+temp1));   
        %% sample nu
    sum1 = sum(log(lam));
    sum2 = sum(1./lam);
    fnu = @(x) T*(x/2.*log(x/2)-gammaln(x/2)) - (x/2+1)*sum1 - x/2*sum2;
    f1 = @(x) T/2*(log(x/2)+1-psi(x/2)) - .5*(sum1+sum2);
    f2 = @(x) T/(2*x) - T/4*psi(1,x/2);
    S = 1;    
    nut = nu;
    while abs(S) > 10^(-5)     % stopping criteria
        S = f1(nut);
        Knu = -f2(nut);        % infomation matrix
        nut = nut + Knu\S;
    end
    sqrtDnu = sqrt(1/Knu);
    nuc = nut + sqrtDnu*randn; 
    if nuc < nuub
        alp = exp(fnu(nuc)-fnu(nu)) ... 
            * normpdf(nu,nut,sqrtDnu)/normpdf(nuc,nut,sqrtDnu);
        if alp > rand
            nu = nuc;
            countnu = countnu+1;
        end    
    end    
    if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end     
    
    if loop>burnin
        i = loop-burnin;
        store_exph(i,:) = exp(h/2)'; 
        store_theta(i,:) = [psi_p nu mu' muh phih sigh2];        
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
exphhat = mean(store_exph)'; 
exphlb = quantile(store_exph,.05)';
exphub = quantile(store_exph,.95)';
tid = linspace(2005,2013,T)';
figure; plot(tid, [exphhat exphlb exphub]);
box off; xlim([2005 2013]);

figure;
subplot(1,2,1); hist(store_theta(:,1),50); box off;
subplot(1,2,2); hist(store_theta(:,2),50); box off;


%% ===== GOLDEN CAPTURE, appended by the savegolden patch; the computation above is untouched =====
golden_named = {};
golden_draws = [who('store_*'); {}'];
golden = struct('matlab', version);
for golden_v = golden_named(:)'
    if exist(golden_v{1}, 'var')
        golden.(golden_v{1}) = eval(golden_v{1});
        fprintf('%s = %s\n', golden_v{1}, mat2str(golden.(golden_v{1}), 8));
    end
end
for golden_v = golden_draws(:)'
    if exist(golden_v{1}, 'var')
        golden_x = eval(golden_v{1});
        if isnumeric(golden_x) && ismatrix(golden_x) && size(golden_x,1) > 1
            golden.([golden_v{1} '_mean']) = mean(golden_x, 1);
            golden.([golden_v{1} '_q05']) = quantile(golden_x, .05, 1);
            golden.([golden_v{1} '_q95']) = quantile(golden_x, .95, 1);
            if size(golden_x, 2) <= 12
                fprintf('%s posterior means: %s\n', golden_v{1}, mat2str(mean(golden_x, 1), 6));
            end
        end
    end
end
save('golden_capture.mat', 'golden');
