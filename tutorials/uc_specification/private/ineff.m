function IF = ineff(draws, L)
% ineff - inefficiency factors of the columns of draws: the long-run variance over
% the variance, 1 + 2*sum_{l=1}^{L} (1 - l/(L+1))*rho_l with Bartlett weights, as
% in the book's inefficiency_factor.m and specvar0.m.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 6.5.2.
[R, k] = size(draws);
IF = zeros(1, k);
for j = 1:k
    x = draws(:,j) - mean(draws(:,j));
    g0 = (x'*x)/R;
    s = g0;
    for l = 1:L
        s = s + 2*(1 - l/(L+1))*(x(1+l:end)'*x(1:end-l))/R;
    end
    IF(j) = s/g0;
end
end
