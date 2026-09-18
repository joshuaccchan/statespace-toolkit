% This script implements the "augmented HP filter" in Grant and Chan (2017)
%
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

%% prior
tau00 = 750; Vtau0 = 100;
b_sigc2 = 3;
phi0 = [1.3 -.7]'; iVphi = speye(2);

R = 50000;
count = 0;
tempphi = repmat(phi0',R,1) + (chol(iVphi\speye(2),'lower')*randn(2,R))';
for i=1:R
    phic = tempphi(i,:)';
    if sum(phic) < .99 && phic(2) - phic(1) < .99 && phic(2) > -.99
        count = count+1;
    end    
end
phi_const = 1/(count/R);
pri_sigc2 = @(x) log(1/b_sigc2) -10^100*(x>b_sigc2);
prior = @(ph,sy,t0) -log(2*pi)+.5*log(det(iVphi))+log(phi_const)-.5*(ph-phi0)'*iVphi*(ph-phi0)...
    + pri_sigc2(sy) - log(2*pi*Vtau0) -.5*sum((t0-tau00).^2)/Vtau0;

disp('Starting HP-AR.... ');
disp(' ' );
start_time = clock; 

% initialize the Markov chain
H2 = speye(T) - 2*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
    + sparse(3:T,1:(T-2),ones(1,T-2),T,T);
H2H2 = H2'*H2;
tau0 = [y(1) y(1)]';
sigc2 = .5;
lam = 1600;
phi = [1.35 -.7]';

Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
    - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);
Xdel = [(2:T+1)' -(1:T)'];
n_grid = 500;

%% initialize for storeage
store_theta = zeros(nsims,5); % [phi', sigc2, tau0]
store_tau = zeros(nsims,T); 
store_mu = zeros(nsims,T-1); 
count_phi = 0;

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for isim = 1:nsims + burnin     
        % sample tau  
    HphiHphi = Hphi'*Hphi;
    alp_tau = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];    
    Ktau = (lam*H2H2 + HphiHphi)/sigc2;    
    tau_hat = Ktau\(lam*H2H2*alp_tau + HphiHphi*y)/sigc2;
    tau = tau_hat + chol(Ktau,'lower')'\randn(T,1);   
    
        % sample phi
    c = y-tau;
    Xphi = [[0;c(1:T-1)] [0;0;c(1:T-2)]];    
    Kphi = iVphi + Xphi'*Xphi/sigc2;
    phi_hat = Kphi\(iVphi*phi0 + Xphi'*c/sigc2);
    flag = 0; count = 0;
    while flag == 0 && count < 100
        phic = phi_hat + chol(Kphi,'lower')'\randn(2,1);
        if sum(phic) < .99 && phic(2) - phic(1) < .99 && phic(2) > -.99
            phi = phic;
            flag = 1;
            count_phi = count_phi + 1;
        end
        count = count + 1;
    end
    Hphi = speye(T) - phi(1)*sparse(2:T,1:(T-1),ones(1,T-1),T,T) + ...
        - phi(2)*sparse(3:T,1:(T-2),ones(1,T-2),T,T);     
        
        % sample sigc2
    u = [Hphi*(y-tau) ...
        tau-2*[tau0(1);tau(1:end-1)]+[tau0(2);tau0(1);tau(1:end-2)]];
    c1 = sum(u(:,1).^2);    
    c2 = sum(u(:,2).^2);
    gy = @(x) -T*log(x) - .5*(c1 + lam*c2)./x;
    sigc2grid = linspace(rand/100,b_sigc2-rand/100,n_grid);
    logpsigc2 = gy(sigc2grid) + pri_sigc2(sigc2grid);    
    psigc2 = exp(logpsigc2-max(logpsigc2));
    psigc2 = psigc2/sum(psigc2);
    cumsumy = cumsum(psigc2);
    sigc2 = sigc2grid(find(rand<cumsumy, 1 ));  
    
        % sample tau0    
    Kdel = diag([1/Vtau0  1/Vtau0]) + lam/sigc2*Xdel'*H2H2*Xdel;
    del_hat = Kdel\([tau00/Vtau0; tau00/Vtau0] + lam/sigc2*Xdel'*H2H2*tau);
    del = del_hat + chol(Kdel,'lower')'\randn(2,1);
    tau0 = del;

    if (mod(isim, 10000) == 0)
        disp([num2str(isim) ' loops... '])
    end         
    
    if isim > burnin
        i = isim - burnin;
        store_tau(i,:) = tau';
        store_theta(i,:) = [phi' sigc2 tau0'];
        store_mu(i,:) = 4*(tau(2:end)-tau(1:end-1))';        
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');        
    [ml mlstd] = ml_HP_AR(y,store_theta,prior,M);    
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
end

tau_hat = mean(store_tau)';
tauCI = quantile(store_tau,[.1 .9])';
theta_hat = mean(store_theta)';
mu_hat = mean(store_mu)';

%% plot of graphs
figure; 
hold on ; box on ;
    area([8 12],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([26 30],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');   
    area([43 46],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([54 57],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([92 96],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([108 113],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([133 135],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([140 144],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([175 177],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([213 216],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([244 250],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    plot(1:T, (y-tau_hat), 'LineWidth',1,'Color','blue');
    plot(1:T, zeros(T,1),'-k','LineWidth',1);
hold off
axis([1, T, -8.2, 6.2]); box off;
set(gca,'XTick',[13 53 93 133 173 213 253]);
set(gca,'XTickLabel',{'1950','1960','1970','1980','1990','2000','2010'});
title('Estimates of the output gap', 'FontSize', 12); 

figure; 
hold on ; box on ;
    area([8 12],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([26 30],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');   
    area([43 46],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([54 57],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([92 96],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([108 113],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([133 135],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([140 144],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([175 177],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([213 216],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([244 250],[4.5 4.5],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    plot(2:T, mu_hat, 'LineWidth',1,'Color','blue');    
hold off
axis([1, T, -.05, 5]); box off;
set(gca,'XTick',[13 53 93 133 173 213 253]);
set(gca,'XTickLabel',{'1950','1960','1970','1980','1990','2000','2010'});
title('Estimates of the annualized trend output growth', 'FontSize', 12);

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end