% This script estimates the M1 model in Chan, Clark and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

    % prior
Vpistar = 100;  % implicitly pistar0 = 0;
Vb = 100;       % implicitly b0 = 0;
psi0 = zeros(q,1); Vpsi = .25^2*speye(q);
mud0 = [0 1]'; Vmud = [.1^2 .1^2]';
rhod0 = [.95 .95]'; Vrhod = [.1^2 .1^2]';
nud0 = 5*ones(2,1); Sd0 = [.01 .001]'.*(nud0-1);
nub0 = 5; Sb0 = .001*(nub0-1);
nuw0 = 5; Sw0 = .01*(nuw0-1);
nuv0 = 5; Sv0 = .01*(nuv0-1);
nun0 = 5; Sn0 = .01*(nun0-1);
lamv0 = log(1); Vlamv = 100;
lamn0 = log(1); Vlamn = 100;
c_psi = 1/(normcdf((1-psi0)/sqrt(Vpsi))-normcdf((-1-psi0)/sqrt(Vpsi)));
lpsipri = @(x) log(c_psi) -.5*log(2*3.14159*Vpsi) -.5*(x-psi0).^2/Vpsi;

    % initialize for storage
store_theta = zeros(nsim,10+q); % [psi' mud' rhod' sigd2' sigb2 sigw2 phiv phin]
store_pistar = zeros(nsim,T); 
store_d = zeros(nsim,2*T);
store_b = zeros(nsim,T); 
store_lamv = zeros(nsim,T);
store_lamn = zeros(nsim,T);
countpsi = 0; countb = 0; countsigb2 = 0; countrhod = zeros(2,1);

    % initialize the Markov chain
sigd2 = .001*ones(2,1);
sigb2 = .01;
sigb = sqrt(sigb2);
sigw2 = .2;
phiv = .1;
phin = .1;
lamv = var(pi)/2*ones(T,1);
lamw = var(z)*ones(T,1);
invLamw = sparse(1:T,1:T,1./lamw);
lamn = var(pi)/2*ones(T,1);
mud = [0 1]';
rhod = [.98 .98]';
shortd = repmat(mud',T,1);
d = reshape(shortd',T*2,1);
b = .3 + .05*rand(T,1);
psi = zeros(q,1);
phihat = psi;
invDpsic = .0001*eye(q);
H =  speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
Hrhod =  speye(2*T) - sparse(2+1:T*2,1:(T-1)*2,repmat(rhod,T-1,1),2*T,2*T);
Hpsi = buildHpsi(psi,T);
options = optimset('Display', 'off') ;
warning off all;

    % MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp(['Starting MCMC for ' model_name '.... ']);
start_time = clock;

for isim = 1:nsim + burnin
  
        % sample pistar     
    Hb = speye(T) - sparse(2:T,1:(T-1),b(2:T),T,T); 
    alppi = Hb\[b(1)*pi0; sparse(T-1,1)];
    HbiLamvHb = Hb'*sparse(1:T,1:T,1./lamv)*Hb;       
    HiLamnH = H'*sparse(1:T,1:T,[1/(Vpistar*lamn(1)); 1./lamn(2:end)])*H;
    ztilde = Hpsi\(z-shortd(:,1));
    tmpXpi = Hpsi\sparse(1:T,1:T,shortd(:,2));
    Xtilde_pi = tmpXpi.*(abs(tmpXpi)>1e-6);
    Kpistar = HbiLamvHb + Xtilde_pi'*Xtilde_pi/sigw2 + HiLamnH;
    pistarhat = Kpistar\(HbiLamvHb*(pi-alppi) + Xtilde_pi'*ztilde/sigw2); % pi*_0 = 0
    pistar = pistarhat + chol(Kpistar,'lower')'\randn(T,1);
    
        % sample b
    [b,flag] = sample_b(b,pi,pistar,lamv,pi0,Vb,sigb2);
    countb = countb + flag;
    
        % sample d    
    Xd = SURform([ones(T,1) pistar]);     
    tmpXd = Hpsi\Xd;    
    Xtilde_d = tmpXd.*(abs(tmpXd)>1e-6);       
    HdiSigdHd = Hrhod'*sparse(1:2*T,1:2*T,[(1-rhod)./sigd2; repmat(1./sigd2,T-1,1)])*Hrhod;
    Kd = HdiSigdHd + Xtilde_d'*Xtilde_d/sigw2;
    deld = Hrhod\[mud; repmat((1-rhod).*mud,T-1,1)];
    CKd = chol(Kd,'lower');    
    dhat = CKd'\(CKd\(HdiSigdHd*deld + Xtilde_d'*(Hpsi\z)/sigw2));
    d = dhat + CKd'\randn(2*T,1); 
    shortd = reshape(d,2,T)';
    
        % sample psi    
    fpsi = @(x) -llike_MAq(x,z-Xd*d,sigw2) - lpsipri(x);
    [psi,flag,psihat,invDpsic] = sample_psi(psi,fpsi,isim,invDpsic,options);
    Hpsi = buildHpsi(psi,T);    
    countpsi = countpsi + flag; 
    
        % sample mud
    for i=1:2
        Dmud = 1/(1/Vmud(i) + ((T-1)*(1-rhod(i))^2 + (1-rhod(i)^2))/sigd2(i));
        mudhat = Dmud*(mud0(i)/Vmud(i) + (1-rhod(i)^2)/sigd2(i)*shortd(1,i) ...
            + (1-rhod(i))/sigd2(i)*sum(shortd(2:end,i)-rhod(i)*shortd(1:end-1,i)));
        mud(i) = mudhat + sqrt(Dmud)*randn;
    end
    
    %% sample rhod
    for i=1:2
        Xrhod = shortd(1:end-1,i)-mud(i);
        yrhod = shortd(2:end,i) - mud(i);
        Drhod = 1/(1/Vrhod(i) + Xrhod'*Xrhod/sigd2(i));
        rhodhat = Drhod*(rhod0(i)/Vrhod(i) + Xrhod'*yrhod/sigd2(i));  
        rhodc = rhodhat + sqrt(Drhod)*randn;
        g = @(x) -.5*log(sigd2(i)./(1-x.^2)) ...
            -.5*(1-x.^2)/sigd2(i)*(shortd(1,i)-mud(i))^2;
        if abs(rhodc) < .98
            alpMH = exp(g(rhodc)-g(rhod(i)));
            if alpMH>rand
                rhod(i) = rhodc;
                countrhod(i) = countrhod(i)+1;                
             end
        end    
    end
    Hrhod =  speye(2*T) - sparse(2+1:T*2,1:(T-1)*2,repmat(rhod,T-1,1),2*T,2*T);
    
        % sample lamv
    Ystar = log((pi-pistar-b.*[pi0; pi(1:T-1)-pistar(1:T-1)]).^2 + .0001);
    loglamv = SVRW(Ystar,log(lamv),phiv,lamv0,Vlamv);    
    lamv = exp(loglamv);    
    
        % sample lamn
    Ystar = log(([pistar(1)/sqrt(Vpistar); pistar(2:end)-pistar(1:end-1)]).^2 + .0001);
    loglamn = SVRW(Ystar,log(lamn),phin,lamn0,Vlamn);
    lamn = exp(loglamn); 

        % sample sigb2
    e2 = (b(2:end) - b(1:end-1)).^2;
    sigb2c = sample_sig2(e2,nub0,Sb0);    
    sigbc = sqrt(sigb2c);
    sigb = sqrt(sigb2);
    alpMH = -sum(log(normcdf((1-b(1:T))/sigbc)-normcdf(-b(1:T)/sigbc))) + ...
        sum(log(normcdf((1-b(1:T))/sigb)-normcdf(-b(1:T)/sigb)));    
    if alpMH > log(rand)
        sigb2 = sigb2c;         
        sigb = sigbc;
        countsigb2 = countsigb2 + 1;
    end
    
        % sample sigd2
    for i=1:2       
        e2 = [(shortd(1,i)-mud(i))*sqrt(1-rhod(i)^2); ...
            shortd(2:end,i)-rhod(i)*shortd(1:end-1,i)-mud(i)*(1-rhod(i))].^2;        
        sigd2(i) = sample_sig2(e2,nud0(i),Sd0(i));
    end   
    
        % sample sigw2
    e2 = (Hpsi\(z - Xd*d)).^2;
    sigw2 = sample_sig2(e2,nuw0,Sw0);    
    
        % sample phiv
    e2 = (loglamv(2:end) - loglamv(1:end-1)).^2;
    phiv = sample_sig2(e2,nuv0,Sv0);   
    
        % sample phin
    e2 = (loglamn(2:end) - loglamn(1:end-1)).^2;
    phin = sample_sig2(e2,nun0,Sn0);    
        
    if isim > burnin
        isave = isim - burnin;
        store_pistar(isave,:) = pistar';
        store_b(isave,:) = b';
        store_d(isave,:) = d';
        store_lamv(isave,:) = lamv';         
        store_lamn(isave,:) = lamn';        
        store_theta(isave,:) = [psi' mud' rhod' sigd2' sigb2 sigw2 phiv phin];
    end
    
    if (mod(isim, 5000) == 0)
        disp([num2str(isim) ' loops... '])
    end 
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

    % plot graphs
pistar_mean = mean(store_pistar)';
b_mean = mean(store_b)';
d_mean = mean(store_d)';
lamv_mean = mean(store_lamv)';
lamn_mean = mean(store_lamn)';
theta_mean = mean(store_theta)';
pistar_CI = quantile(store_pistar,[.16 .84]);
lamv_CI = quantile(store_lamv,[.16 .84]);
lamn_CI = quantile(store_lamn,[.16 .84]);
b_CI = quantile(store_b,[.16 .84]);
d_CI = quantile(store_d,[.16 .84]);

tid = linspace(t0(1),t0(2),T)';
figure;
    % plot trend inflation
subplot(1,2,1); 
hold on        
    plot(tid, pistar_mean, 'LineWidth',1,'Color','black');        
    plot(tid, Einf(2:end), 'LineWidth',1,'Color','blue');
    plot(tid, pi, 'LineWidth',1 ,'Color','red');     
hold off    
box off; xlim([t0(1)-.5 t0(2)+.5]);
legend('\pi^*_t','z_t', '\pi_t','Location','NorthEast');
subplot(1,2,2); 
hold on        
    plotCI(tid,pistar_CI(1,:)',pistar_CI(2,:)');
    h1 = plot(tid, pistar_mean, 'LineWidth',1,'Color','black');        
    h2 = plot(tid, Einf(2:end), 'LineWidth',1,'Color','blue');    
hold off    
box off; xlim([t0(1)-.5 t0(2)+.5]);    
legend([h1 h2], '\pi^*_t','z_t','Location','NorthEast');
set(gcf,'Position',[200 200 800 300]);

    % plot d_0t and d_1t
figure;
subplot(1,2,1); 
hold on            
    plotCI(tid,d_CI(1,1:2:end)',d_CI(2,1:2:end)'); 
    plot(tid, d_mean(1:2:end), 'LineWidth',1,'Color','black');    
hold off
box off; xlim([t0(1)-.5 t0(2)+.5]);
title('d_{0t}','FontSize',12); 
subplot(1,2,2); 
hold on            
    plotCI(tid,d_CI(1,2:2:end)',d_CI(2,2:2:end)'); 
    plot(tid, d_mean(2:2:end), 'LineWidth',1,'Color','black');    
hold off
box off; xlim([t0(1)-.5 t0(2)+.5]);
title('d_{1t}','FontSize',12); 
set(gcf,'Position',[200 200 800 300]);

    % plot lambda_v and lambda_n
figure;
subplot(1,2,1); 
hold on        
    plotCI(tid,lamv_CI(1,:)',lamv_CI(2,:)');    
    plot(tid, lamv_mean, 'LineWidth',1,'Color','black');    
hold off
box off; xlim([t0(1)-.5 t0(2)+.5]); title('\lambda_{v,t}','FontSize',12);
subplot(1,2,2); 
hold on        
    plotCI(tid,lamn_CI(1,:)',lamn_CI(2,:)'); 
    plot(tid, lamn_mean, 'LineWidth',1,'Color','black');    
hold off
title('\lambda_{n,t}','FontSize',12); box off; xlim([t0(1)-.5 t0(2)+.5]);
set(gcf,'Position',[200 200 800 300]);

figure
hold on        
    plotCI(tid,b_CI(1,:)',b_CI(2,:)');
    plot(tid, b_mean, 'LineWidth',1,'Color','black');    
hold off 
box off; xlim([t0(1)-.5 t0(2)+.5]);  ylim([0 1]);
title('b_t','FontSize',12);    

    
