% This script estimates the SV-MA model.
% See:
%
% Chan, J.C.C. and Grant, A.L. (2016). Modeling Energy Price Dynamics:
% GARCH versus Stochastic Volatility, Energy Economics, 54, 182-189.

    %% prior
phih0 = .97; Vphih = .1^2;
mu0 = 0; Vmu = 10;
muh0 = 1; Vmuh = 10;
psi0 = 0; Vpsi = 1;
nuh = 5; Sh = .2^2*(nuh-1);

phih_const = 1/(normcdf(1,phih0,sqrt(Vphih))-normcdf(-1,phih0,sqrt(Vphih)));
psi_const = 1/(normcdf(1,psi0,sqrt(Vpsi))-normcdf(-1,psi0,sqrt(Vpsi)));
prior = @(m,p,mh,ph,oh) -.5*log(2*pi*Vmu) -.5*(m-mu0)^2/Vmu ...
    -.5*log(2*pi*Vpsi) + log(psi_const) - .5*(p-psi0)^2/Vpsi ... 
    -.5*log(2*pi*Vmuh) - .5*(mh-muh0)^2/Vmuh ... 
    -.5*log(2*pi*Vphih) + log(phih_const) -.5*(ph-phih0)^2/Vphih ...
    + nuh*log(Sh) - gammaln(nuh) - (nuh+1)*log(oh) - Sh/oh;

    %% initialize the Markov chain
psi = 0;
mu = mean(y);
muh = log(var(y)); phih = .98; 
omegah2 = .2^2;
h = muh + sqrt(omegah2)*randn(T,1);

    %% initialize for storage
store_theta = zeros((nloop - burnin),5); % [mu psi muh phih omegah2]
store_h = zeros((nloop - burnin),T);
store_Q = zeros(nloop - burnin,2);

    %% construct a few things
newnuh = T/2 + nuh;
Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
X = ones(T,1);
counth = 0; countpsi = 0; countphih = 0;

disp('Starting SV-MA.... ');
disp(' ' );
start_time = clock;    

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop
        %% sample mu    
    Xtilde = Hpsi\X;    ytilde = Hpsi\y;
    temp1 = Xtilde'*sparse(1:T,1:T,exp(-h));
    Dmu = 1/(temp1*Xtilde + 1/Vmu);
    muhat = Dmu*(mu0/Vmu + temp1*ytilde);
    mu = muhat + sqrt(Dmu)*randn;        
    
        %% sample h    
    HiSH = Hphi'*spdiags([(1-phih^2)/omegah2; 1/omegah2*ones(T-1,1)],0,T,T)*Hphi;
    deltah = Hphi\[muh; muh*(1-phih)*ones(T-1,1)];
    HiSHdeltah = HiSH*deltah;
    s2 = (Hpsi\(y-mu)).^2;
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
        counth = counth + 1;
    end     
        %% sample omegah2
    errh = [(h(1)-muh)*sqrt(1-phih^2);  h(2:end)-phih*h(1:end-1)-muh*(1-phih)];
    newSh = Sh + sum(errh.^2)/2;
    omegah2 = 1/gamrnd(newnuh, 1./newSh);    
        %% sample phih
    Xphih = h(1:end-1)-muh;
    yphih = h(2:end) - muh;
    Dphih = 1/(1/Vphih + Xphih'*Xphih/omegah2);
    phihhat = Dphih*(phih0/Vphih + Xphih'*yphih/omegah2);
    phihc = phihhat + sqrt(Dphih)*randn;
    g = @(x) -.5*log(omegah2./(1-x.^2))-.5*(1-x.^2)/omegah2*(h(1)-muh)^2;
    if abs(phihc)<.9999
        alpMH = exp(g(phihc)-g(phih));
        if alpMH>rand
            phih = phihc;
            Hphi = speye(T) - sparse(2:T,1:(T-1),phih*ones(1,T-1),T,T);
            countphih = countphih+1;
        end
    end    
        %% sample muh    
    Dmuh = 1/(1/Vmuh + ((T-1)*(1-phih)^2 + (1-phih^2))/omegah2);
    muhhat = Dmuh*(muh0/Vmuh + (1-phih^2)/omegah2*h(1) + (1-phih)/omegah2*sum(h(2:end)-phih*h(1:end-1)));
    muh = muhhat + sqrt(Dmuh)*randn;    
    
    %% sample psi
    f = @(x) fMA1(x,y-mu,h) + .5*(psi0-x)^2/Vpsi;  % negative of the log-density
    psihat = fminbnd(f,-.98,.98);
    sqVpsic = .04; Vpsic = sqVpsic^2;
    psic = psihat + sqVpsic*randn;
    if abs(psic)<.99
        alpMH = -f(psic) + f(psi) ...
            -.5*(psi-psihat)^2/Vpsic + .5*(psic-psihat)^2/Vpsic;
    else
        alpMH = -inf;
    end
    if alpMH>log(rand)
        psi = psic;
        Hpsi = speye(T) + sparse(2:T,1:(T-1),psi*ones(1,T-1),T,T); 
        countpsi = countpsi + 1;
    end     
    if ( mod( loop, 5000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end  
    
    if loop>burnin
        i=loop-burnin;
        store_h(i,:)  = h';         
        store_theta(i,:) = [mu psi muh phih omegah2];         
        
            % compute Q stats 
        u = (Hpsi\(y-mu))./exp(h/2);
        rtmp = autocorr(u,nlag);
        Q = T*((T+2)./(T-(1:nlag)))*rtmp(2:end).^2;
        rtmp = autocorr(u.^2,nlag);
        Q2 = T*((T+2)./(T-(1:nlag)))*rtmp(2:end).^2;
        store_Q(i,:) = [Q Q2];
    end

end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

hhat = mean(exp(store_h/2))';  %% plot std dev
thetahat = mean(store_theta)';
Qhat = mean(store_Q)';
Qstd = std(store_Q)';
thetastd = std(store_theta)';
accept = [counth countpsi countphih]/nloop;

figure;    
plot(tid, hhat, 'LineWidth',2,'Color','black'); box off;
title('exp(h_t/2)');

%% compute ml
if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');    
    [ml mlstd] = ml_sv_MA(y,store_theta,hhat,prior,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );    
end

fprintf('\n'); 
fprintf('Parameter   | Posterior mean (Posterior std. dev.):\n'); 
fprintf('mu          | %.2f (%.2f)\n', thetahat(1), thetastd(1)); 
fprintf('mu_h        | %.2f (%.2f)\n', thetahat(3), thetastd(3)); 
fprintf('phi_h       | %.2f (%.2f)\n', thetahat(4), thetastd(4)); 
fprintf('omega2_h    | %.2f (%.2f)\n', thetahat(5), thetastd(5)); 
fprintf('psi         | %.2f (%.2f)\n', thetahat(2), thetastd(2)); 

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end


