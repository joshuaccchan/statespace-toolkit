function den = lmvnpdf(X, mu, Sig)
n = length(mu);
CSig = chol(Sig,'lower');
e = CSig\(X-mu);
den = - n/2*log(2*pi) - sum(log(diag(CSig))) - .5*(e'*e);
end