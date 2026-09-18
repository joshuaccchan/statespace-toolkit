% this function returns N draws from the normal distribution with mean mu and
% variance sigma2, truncated on (a, b)
%
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

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