% % =======================================================================
% % support function: this function returns N draws from a truncated normal
% % distribution with mean mu and variance sigma2 
% % a and b are the points of truncations
% %
% % See Chan, J.C.C. and Eisenstat, E. (2015). "Marginal Likelihood Estimation
% % with the Cross-Entropy Method," Econometric Reviews, 34(3), 256-285.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

function t = tnormrnd(mu,sigma2,a,b,N)

if ( nargin < 4  )
    error( 'wrong # of arguments' );
end

K = length( mu );

if ( nargin < 5  )
     N = K;
end

if ( ( K ~= N ) || ( length( sigma2 ) ~= N ) ) && ( ( K ~= 1 ) )
    error( 'dimensions of mu and sigma must equal N')
end
    
if K == 1
    mu = ones(N,1)*mu;
    sigma2 = ones(N,1)*sigma2;
end

sigma = sqrt(sigma2);
u = rand(N,1);
p1 = normcdf((a-mu)./sigma);
p2 = normcdf((b-mu)./sigma);
C = norminv(p1+(p2-p1).*u);
t = mu + sigma.*C;
