% This script implements the HP filter using the band matrix algorithm in 
% Grant and Chan (2017)
% The two initial values are treated as parameters to be estimated
%
% See:
% Grant, A.L. and Chan, J.C.C. (2017). Reconciling output gaps: Unobserved
% components model and Hodrick-Prescott filter, Journal of Economic Dynamics
% and Control, 75, 114-121.

%% prior
tau00 = 750; Vtau0 = 100;
sigy2_ub = 3;
pri_sigy2 = @(x) log(1/sigy2_ub);
prior = @(sy,t0) pri_sigy2(sy) -log(2*pi*Vtau0) - .5*sum((t0-tau00).^2)/Vtau0;

disp('Starting MCMC for HP.... ');
disp(' ' );
start_time = clock; 

% initialize the Markov chain
H2 = speye(T) - 2*sparse(2:T,1:(T-1),ones(1,T-1),T,T) ...
    + sparse(3:T,1:(T-2),ones(1,T-2),T,T);
H2H2 = H2'*H2;
tau0 = [y(1) y(1)]';
sigy2 = .5;
lam = 1600;
ngrid = 500;
Xdel = [(2:T+1)' -(1:T)'];
 
%% initialize for storeage
store_theta = zeros(nsims,3); % [sigy2, tau0]
store_tau = zeros(nsims,T); 
store_mu = zeros(nsims,T-1); 
countphi = 0;

rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for isim = 1:nsims+burnin
     
    %% sample tau  
    alp = H2\[2*tau0(1)-tau0(2);-tau0(1);sparse(T-2,1)];    
    Ktau = lam*H2H2 + speye(T);    
    tauhat = Ktau\(lam*H2H2*alp + y);
    tau = tauhat + chol(Ktau,'lower')'\randn(T,1);   
        
    %% sample sigy2
    u = [y-tau H2*(tau-alp)];
    c1 = sum(u(:,1).^2);
    c2 = u(:,1)'*u(:,2); 
    c3 = sum(u(:,2).^2);
    gy = @(x) -T*log(x) - .5*lam*c3./x - 1./(2*x)*c1;
    sigy2grid = linspace(rand/100,sigy2_ub-rand/100,ngrid);
    logpsigy2 = gy(sigy2grid) + pri_sigy2(sigy2grid);    
    psigy2 = exp(logpsigy2-max(logpsigy2));
    psigy2 = psigy2/sum(psigy2);
    cumsumy = cumsum(psigy2);
    sigy2 = sigy2grid(find(rand<cumsumy, 1 ));
    
    %% sample tau0
    uy = y-tau;
    Kdel = diag([1/Vtau0  1/Vtau0]) + lam/sigy2*Xdel'*H2H2*Xdel;
    delhat = Kdel\([tau00/Vtau0; tau00/Vtau0] + lam/sigy2*Xdel'*H2H2*tau);
    del = delhat + chol(Kdel,'lower')'\randn(2,1);
    tau0 = del;

    if (mod(isim, 10000) == 0)
        disp([num2str(isim) ' loops... '])
    end     
    
    if isim>burnin
        i = isim-burnin;
        store_tau(i,:) = tau';
        store_theta(i,:) = [sigy2 tau0'];
        store_mu(i,:) = 4*(tau(2:end)-tau(1:end-1))';
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

if cp_ml
    start_time = clock;
    disp('Computing the marginal likelihood.... ');        
    [ml mlstd] = ml_HP(y,store_theta,prior,lam,M);
    disp( ['ML computation takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
end

tauhat = mean(store_tau)';
tauCI = quantile(store_tau,[.1 .9])';
thetahat = mean(store_theta)';
muhat = mean(store_mu)';

%% plot of graphs
figure; 
hold on ; box on ;
    area([8 12],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([27 30],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');   
    area([43 46],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([54 57],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([92 96],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([108 113],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([133 135],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([139 144],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([175 177],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([217 220],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([244 250],[6 6],-8,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    plot(1:T, (y-tauhat), 'LineWidth',1,'Color','blue');
    plot(1:T, zeros(T,1),'-k','LineWidth',1);
hold off
axis([1, T, -8.2, 6.2]); box off;
set(gca,'XTick',[13 53 93 133 173 213 253]);
set(gca,'XTickLabel',{'1950','1960','1970','1980','1990','2000','2010'});
title('Estimates of the output gap', 'FontSize', 12); 

figure; 
hold on ; box on ;
    area([8 12],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([27 30],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');   
    area([43 46],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([54 57],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([92 96],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([108 113],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([133 135],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([139 144],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([175 177],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([217 220],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    area([244 250],[6 6],0,'FaceColor',[.91 .91 .91],'EdgeColor','none'); set(gcf,'renderer','painters');
    plot(2:T, muhat, 'LineWidth',1,'Color','blue');    
hold off
axis([1, T, -.05, 6]); box off;
set(gca,'XTick',[13 53 93 133 173 213 253]);
set(gca,'XTickLabel',{'1950','1960','1970','1980','1990','2000','2010'});
title('Estimates of the annualized trend output growth', 'FontSize', 12);

if cp_ml
    fprintf('\n'); 
    fprintf('log marginal likelihood: %.1f (%.2f)\n', ml, mlstd); 
end