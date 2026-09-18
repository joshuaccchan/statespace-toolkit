% ssm.tnormrnd - N draws from the normal truncated to (a, b), by inverse
% transform.
%
%   t = ssm.tnormrnd(mu, sigma2, a, b)
%   t = ssm.tnormrnd(mu, sigma2, a, b, N)
%
%   mu, sigma2 : mean and VARIANCE of the untruncated normal, either scalars,
%                which are expanded to N identical values, or length-N vectors
%   a, b       : truncation bounds, scalar or length-N
%   N          : number of draws (default length(mu))
%   t          : N x 1 vector of draws
%
% Code-identical to bvar.util.tnormrnd in bvar-toolkit, and the two must stay so
% (tests/unit/test_twins.m).

function t = tnormrnd(mu, sigma2, a, b, N)

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