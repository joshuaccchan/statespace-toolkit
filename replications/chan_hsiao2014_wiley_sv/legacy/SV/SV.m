% % =======================================================================
% % univariate stationary AR(1) stochastic volatility
% % See Chan, J.C.C. and Hsiao, C.Y.L (2014). Estimation of Stochastic
% % Volatility Models with Heavy Tails and Serial Dependence. 
% % In: I. Jeliazkov and X.S. Yang (Eds.), Bayesian Inference in the 
% % Social Sciences, 159-180, John Wiley & Sons, New York.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function [h mu phi sig2 S] = SV(Ystar,h,mu,phi,sig2,prior)

mu0 = prior(1);  invVmu = prior(2);
phi0 = prior(3); invVphi = prior(4);
nu0 = prior(5);  S0 = prior(6);
T = length(h);
%% normal mixture
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

%% sample S from a 7-point distrete distribution
temprand = rand(T,1);
q = repmat(pi,T,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,T,1), repmat(sqrtsigi,T,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;
    
%% sample h using the precision-based algorithm
% y* = h + d + epsilon*, epsilon* ~ N(0,Sig_y*),
% Hphi h = alpha-tilde + zeta, zeta ~ N(0,Sig_h),
% where 
% d_t = E(epsilon*_t), Sig_y* = diag(var(epsilon*_1), ..., epsilon*_T),
% Sig_h = diag(sig2/(1-phi^2), sig2, ..., sig2)
Hphi = speye(T)-sparse(2:T,1:(T-1),phi*ones(1,T-1),T,T);
invSigh = sparse(1:T,1:T,[(1-phi^2)/sig2; 1/sig2*ones(T-1,1)]);
d = mi(S)'; invSigystar = spdiags(1./sigi(S)',0,T,T);
alpha = Hphi\[mu; ((1-phi)*mu)*ones(T-1,1)];
Kh = Hphi'*invSigh*Hphi + invSigystar;
Ch = chol(Kh,'lower');
hhat = Kh\(Hphi'*invSigh*Hphi*alpha + invSigystar*(Ystar-d));
h = hhat + Ch'\randn(T,1);

%% sample sig2
newS = S0 + sum([(h(1)-mu)*sqrt(1-phi^2); h(2:end)-phi*h(1:end-1)-mu*(1-phi)].^2)/2;
sig2 = 1/gamrnd(nu0+T/2,1/newS);
     
%% sample phi
Xphi = h(1:end-1)-mu;
zphi = h(2:end) - mu;
Dphi = 1/(invVphi + Xphi'*Xphi/sig2);
phihat = Dphi*(invVphi*phi0 + Xphi'*zphi/sig2);
phic = phihat + sqrt(Dphi)*randn;
g = @(x) .5*log(1-x.^2)-.5*(1-x.^2)/sig2*(h(1)-mu)^2;
if abs(phic)<.9999
    alp = exp(g(phic)-g(phi));
    if alp>rand
        phi = phic;
    end
end 
    
%% sample mu    
Dmu = 1/(invVmu + ((T-1)*(1-phi)^2 + (1-phi^2))/sig2);
muhat = Dmu*(invVmu*mu0 + ...
    (1-phi^2)/sig2*h(1) + (1-phi)/sig2*sum(h(2:end)-phi*h(1:end-1)));
mu = muhat + sqrt(Dmu)*randn;

end
