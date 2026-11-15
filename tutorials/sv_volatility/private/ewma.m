function v = ewma(y, lam)
% ewma - the RiskMetrics variance of the day after y(end), with mean zero:
% s2_{k+1} = lam*s2_k + (1 - lam)*y_k^2, started at s2_1 = mean(y.^2), the mean squared
% return of all of y
s = filter(1 - lam, [1 -lam], y.^2, lam*mean(y.^2));
v = s(end);
end
