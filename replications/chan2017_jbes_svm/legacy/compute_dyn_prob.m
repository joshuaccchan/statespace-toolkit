% % This script computes the dynamic probabilities that alpha_t not equal 0
% %
% % See:
% % Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with
% % Time-Varying Parameters: An Application to Inflation Modeling, 
% % Journal of Business and Economic Statistics, 35(1), 17-28.

postalp = zeros(T,1);
prialp = zeros(T,1);

Omega = zeros(2,2);
disp('Computing the dynamic probabilities that alpha_t not equal 0.... ');
% evaluate the posterior densities of alpha_t at 0
for loop = 1:nloop - burnin
    Omega([1 2 4]) = store_theta(loop,5:7);
    Omega(3) = store_theta(loop,6);
    h = store_h(loop,:)';
    
    invOmegagam = [invVgam sparse(2,2*(T-1)); ...
        sparse(2*(T-1),2) kron(speye(T-1),Omega\speye(2))];    
    Xgam = SURform([exp(h) ones(T,1)]); 
    tmp1 =  Xgam' * sparse(1:T,1:T,1./exp(h));
    Kgam = H'*invOmegagam*H + tmp1*Xgam;
    Dgam = diag(Kgam\speye(2*T));         
    gamhat = Kgam\(tmp1*y);
    postalp = postalp + normpdf(0,gamhat(1:2:end),sqrt(Dgam(1:2:end))); 
end

            
% evaluate the prior densities of alpha_t at 0
N = 50000;
H = speye(T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
for ii=1:N
    Omega = iwishrnd(SOmega,nuOmega);       
    omega2alp = Omega(1);
    invOmegaalp = sparse(1:T,1:T,[1/Vgam(1) 1/omega2alp*ones(1,T-1)]);
    Kalp = H'*invOmegaalp*H;
    Dalp = diag(Kalp\speye(T));         
    prialp = prialp + normpdf(0,0,sqrt(Dalp));
end
BF = postalp/(nloop-burnin) ./ (prialp/N);
prob = 1./(1+BF);
