% this function returns N draws from the normal distribution with mean mu 
% and variance sigma2, truncated on (a, b)
%
% See:
% Chan, J.C.C., Koop, G. and Potter, S.M. (2016). A Bounded Model of Time 
% Variation in Trend Inflation, NAIRU and the Phillips Curve, Journal of 
% Applied Econometrics, 31(3), 551-565.

function t = tnormrnd( mu, sigma2, a, b, N)

if ( nargin < 4  )
    error( 'wrong # of arguments' );
end

K = length( mu );

if ( nargin < 5  )
     N = K;
end

if ( ( K ~= N ) | ( length( sigma2 ) ~= N ) ) & ( ( K ~= 1 ) )
    error( 'dimensions of mu and sigma must equal N')
end
    
if K == 1
    mu = ones( N, 1 ) * mu;
    sigma2 = ones( N, 1 ) * sigma2;
end

sigma = sqrt( sigma2 );
u = rand(N,1);
p1 = normcdf( ( a - mu ) ./ sigma );
p2 = normcdf( ( b - mu ) ./ sigma );
C = norminv( p1 + ( p2 - p1 ) .* u );
t = mu + sigma .* C;
end