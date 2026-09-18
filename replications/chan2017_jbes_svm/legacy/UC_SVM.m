% % =======================================================================
% % Unobserved components model with stochastic volatility in mean + TVP
% %
% % y_t    = tau_t + alpha_t exp(h_t) + e_t,            e_t ~ N(0,exp(h_t)),
% % h_t    = mu + phi (h_{t-1}-mu) + beta y_{t-1}+ v_t, v_t ~ N(0,sig2),
% % 
% % gam_t = (alpha_t, tau_t)'
% % gam_t = gam_{t-1} + w_t, w_t ~ N(0,Omega)
% % where Omega is a full matrix
% % 
% % Initial conditions: tau_1 ~ N(0,Vtau), h_1 ~ N(mu,sig2/(1-phi^2)).
% % 
% % Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with
% % Time-Varying Parameters: An Application to Inflation Modeling, 
% % Journal of Business and Economic Statistics, 35(1), 17-28.
% %
% % Please report any errors to joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
nloop = 55000;
burnin = 5000;
is_dym_prob = false;  % true: compute dynamic probabilities
is_alp_const = false; % true: restrict alpha_t to be time-invariant
load 'USCPI.csv';
y = USCPI;
T = length(y);
tid = linspace(1948,2013.25,T)'; 

if is_alp_const && is_alp_const
    error('For UC-SVM-const dynamic probabilities cannot be computed');    
end

    %% prior
phi0 = .97; Vphi = .1^2;
mu0 = 0; Vmu = 10;
Vgam = 10*eye(2); invVgam = Vgam\speye(2);
beta0 = 0; Vbeta = 10;
nuOmega = 10;
if is_alp_const    
    SOmega = (nuOmega+3)*diag([.000001^2 .25^2]);
    model_name = 'UC-SVM-const';
else    
    SOmega = (nuOmega+3)*diag([.1^2 .25^2]);
    model_name = 'UC-SVM';
end
nuh = 10; Sh = .2^2*(nuh-1);
    %% initialize the Markov chain
beta = 0;
mu = log(var(y)); phi = .98; sig2 = .2^2;
Omega = diag([.1^2 .25^2]);
h = mu + sqrt(sig2)*randn(T,1);
exph = exp(h);
    %% initialize for storeage
store_theta = zeros(nloop - burnin,7); % [mu beta phi sig2 Omega([1 2 4])]
store_tau = zeros(nloop - burnin,T); 
store_alp = zeros(nloop - burnin,T); 
store_h = zeros(nloop - burnin,T);
    %% compute a few things outside the loop
H = speye(2*T) - sparse(3:2*T,1:2*(T-1),ones(1,2*(T-1)),2*T,2*T);    
Hphi = speye(T) - sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T);
newnuh = T/2 + nuh;
counth = 0; countlam = 0;
disp(['Starting MCMC for ' model_name '.... ']);
start_time = clock;
for loop = 1:nloop 
        %% sample gam    
    invOmegagam = [invVgam sparse(2,2*(T-1)); ...
        sparse(2*(T-1),2) kron(speye(T-1),Omega\speye(2))];    
    Xgam = SURform([exph ones(T,1)]); 
    temp1 =  Xgam' * sparse(1:T,1:T,1./exph);
    Kgam = H'*invOmegagam*H + temp1*Xgam;
    Cgam = chol(Kgam,'lower');         
    gamhat = Kgam\(temp1*y);
    gam = gamhat + Cgam'\randn(2*T,1);
    alp = gam(1:2:end);
    tau = gam(2:2:end);    

        %% sample h    
    HinvSH = Hphi'*sparse(1:T,1:T,[(1-phi^2)/sig2; 1/sig2*ones(T-1,1)])*Hphi;
    deltah = Hphi\([mu; mu*(1-phi)*ones(T-1,1)] + [0;y(1:end-1)]*beta);
    HinvSHdeltah = HinvSH*deltah;
    s2 = (y-tau).^2;
    errh = 1; ht = h;
    while errh> 10^(-3)
        expht = exp(ht);
        sinvexpht = s2./expht;
        alp2expht = alp.^2.*expht;
        fh = -.5 + .5*sinvexpht - .5*alp2expht;
        Gh = .5*sinvexpht + .5*alp2expht;
        Kh = HinvSH + spdiags(Gh,0,T,T);
        newht = Kh\(fh+Gh.*ht+HinvSHdeltah);
        errh = max(abs(newht-ht));
        ht = newht;          
    end 
    cholHh = chol(Kh,'lower');
    % AR-step:     
    hstar = ht;
    uh = hstar-deltah;
    logc = -.5*uh'*HinvSH*uh -.5*sum(hstar) + ...
        - .5*exp(-hstar)'*(y-tau-alp.*exp(hstar)).^2 + log(3);
    flag = 0;
    while flag == 0
        hc = ht + cholHh'\randn(T,1);
        vhc = hc-ht;
        uhc = hc-deltah;
        alpARc = -.5*uhc'*HinvSH*uhc -.5*sum(hc) + ...
            -.5*exp(-hc)'*(y-tau-alp.*exp(hc)).^2 + .5*vhc'*Kh*vhc - logc;            
        if alpARc > log(rand)
            flag = 1;
        end
    end        
    % MH-step
    vh = h-ht;
    uh = h-deltah;
    alpAR = -.5*uh'*HinvSH*uh -.5*sum(h) + ...
        -.5*exp(-h)'*(y-tau-alp.*exp(h)).^2 + .5*vh'*Kh*vh - logc;
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
    
        %% sample beta    
    ybeta = h(2:end) - mu - phi*(h(1:end-1)-mu);
    Dbeta = 1/(1/Vbeta + y(1:end-1)'*y(1:end-1)/sig2);
    betahat = Dbeta*(beta0/Vbeta + y(1:end-1)'*ybeta/sig2);
    beta = betahat + sqrt(Dbeta)*randn;
    
        %% sample Omega
    err = reshape(gam(3:end,:) - gam(1:end-2,:),2,T-1)';    
    Omega = iwishrnd(SOmega + err'*err,nuOmega + T-1);
    
        %% sample sig2
    errh = [(h(1)-mu)*sqrt(1-phi^2); h(2:end)-phi*h(1:end-1)-mu*(1-phi)-y(1:end-1)*beta];    
    newSh = Sh + sum(errh.^2)/2;    
    sig2 = 1/gamrnd(newnuh, 1./newSh);     
    
        %% sample mu and phi jointly    
    sum1 = sum(h(2:end));
    sum2 = sum(h(1:end-1));    
    flam = @(x) .5*log(1-x(2)^2) - (1-x(2)^2)/(2*sig2)*(h(1)-x(1))^2 ...
        - 1/(2*sig2)*sum((h(2:end) - x(2)*h(1:end-1) - x(1)*(1-x(2))).^2)...
        - 1/(2*Vphi)*(x(2)-phi0)^2 - 1/(2*Vmu)*(x(1)-mu0)^2;
    [lamc,g] = proplam(h,sig2);
    MHprob = flam(lamc) - flam([mu; phi]) + g([mu; phi]) - g(lamc);        
    if exp(MHprob) > rand
        mu = lamc(1);
        phi = lamc(2);
        countlam = countlam+1;
    end
    Hphi = speye(T) - sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T);
    
    if loop>burnin
        i = loop-burnin;
        store_tau(i,:) = tau';
        store_h(i,:) = h'; 
        store_alp(i,:) = alp';        
        store_theta(i,:) =  [mu beta phi sig2 Omega([1 2 4])];  
    end    
    if (mod(loop,10000) == 0)
        disp([num2str(loop) ' loops... '])
    end    
end
clc;
disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );

tauhat = mean(store_tau)';
alphat = mean(store_alp)';
hhat = mean(store_h)';
thetahat = mean(store_theta)';
hCI = quantile(store_h,[.05 .95])';
alpCI = quantile(store_alp,[.05 .95])';
thetaCI = quantile(store_theta,[.05 .95])';

if is_dym_prob
    compute_dyn_prob;    
    figure
    plot(tid,prob,'LineWidth',2,'Color','black');
    box off; ylim([0 1.02]);
    xlim([min(tid)-1 2014]); title('Dynamic Probabilities that \alpha_t \neq 0');
end

figure    
plot(tid, y); title('y_t'); box off; xlim([min(tid)-1 2014]);
set(gcf,'Position',[100 100 500 400]);
    
figure;    
subplot(1,2,1);
hold on    
    plot(tid, hhat, 'LineWidth',2,'Color','black');  
    plot(tid, hCI,'--r','LineWidth',2);  
hold off    
xlim([min(tid)-1 2014]);  title('h_t');
subplot(1,2,2);
hold on    
    plot(tid, alphat, 'LineWidth',2,'Color','black');  
    plot(tid, alpCI,'--r','LineWidth',2);  
hold off      
xlim([min(tid)-1 2014]); title('\alpha_t');
set(gcf,'Position',[100 100 1000 400]);   

apt_rate = [counth countlam]/nloop;

fprintf('\n'); 
fprintf('Parameter         | Posterior mean | 90%% credible interval:\n'); 
fprintf('mu                | %.3f          | (%.3f, %.3f)\n', thetahat(1), thetaCI(1,1), thetaCI(1,2)); 
fprintf('beta              | %.3f          | (%.3f, %.3f)\n', thetahat(2), thetaCI(2,1), thetaCI(2,2)); 
fprintf('phi               | %.3f          | (%.3f, %.3f)\n', thetahat(3), thetaCI(3,1), thetaCI(3,2)); 
fprintf('sigma2            | %.3f          | (%.3f, %.3f)\n', thetahat(4), thetaCI(4,1), thetaCI(4,2)); 
fprintf('omega2_alpha      | %.3f          | (%.3f, %.3f)\n', thetahat(5), thetaCI(5,1), thetaCI(5,2)); 
fprintf('omega_{alpha,tau} | %.3f          | (%.3f, %.3f)\n', thetahat(6), thetaCI(6,1), thetaCI(6,2));
fprintf('omega2_tau        | %.3f          | (%.3f, %.3f)\n', thetahat(7), thetaCI(7,1), thetaCI(7,2));


