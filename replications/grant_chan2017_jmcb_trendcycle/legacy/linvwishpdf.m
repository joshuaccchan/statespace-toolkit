function den = linvwishpdf( W, nu, S )
K = size(S,1);
sum1 = sum(gammaln((nu + 1 - (1:K)) / 2));
const = nu*K/2*log(2) + K*(K-1)/4*log(pi) + sum1;
den = -const + nu/2*log(det(S)) - (nu+K+1)/2*log(det(W)) - trace(S/W)/2;
end