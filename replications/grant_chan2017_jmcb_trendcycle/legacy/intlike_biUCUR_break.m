function llike = intlike_biUCUR_break(y,mu,phi,Sig,tau0,t0)
T = length(y)/2;
d0 = ((1:T)'<t0);
d1 = ((1:T)'>=t0);
y1 = y(1:2:end);
y2 = y(2:2:end);
H = speye(2*T) - sparse(3:T*2,1:(T-1)*2,ones(1,2*(T-1)),2*T,2*T);
Hphi = speye(2*T) - sparse(3:T*2,1:(T-1)*2,repmat(phi(1:2),1,T-1),2*T,2*T) ...
    -sparse(5:T*2,1:(T-2)*2,repmat(phi(3:4),1,T-2),2*T,2*T);
alp = H\(kron(d0,mu(1:2))+ kron(d1,mu(3:4)) +[tau0;sparse(2*(T-1),1)]);
tmp1 = Sig(1:2,3:4)/Sig(3:4,3:4);
tmp2 = Sig(1:2,1:2) - tmp1*Sig(1:2,3:4)';    
a = -kron(speye(T),tmp1)*(H*alp);
B = Hphi + kron(speye(T),tmp1)*H;
P = sparse(1:T,1:2:2*T,ones(T,1),T,2*T);
Lam = Hphi'*kron(speye(T),tmp2\speye(2))*Hphi;
Lam_11 = Lam(1:2:end,1:2:end);
Lam_12 = Lam(1:2:end,2:2:end);
d = P*Lam*(Hphi\a)-Lam_12*y2;
D = P*Hphi'*kron(speye(T),tmp2\speye(2))*B;
HiStauH = H'*kron(speye(T),Sig(3:4,3:4)\speye(2))*H;
DiLam_11= D'/Lam_11;
Ktau = HiStauH + DiLam_11*D;
CKtau = sparse(chol(Ktau,'lower'));
tauhat = CKtau'\(CKtau\(HiStauH*alp + D'*(y1-Lam_11\d)));
tmpu = CKtau'*tauhat;
llike = -T/2*log(2*pi)-T/2*log(det(Sig(3:4,3:4))) ...
    +.5*2*sum(log(diag(chol(Lam_11))))-.5*2*sum(log(diag(CKtau)))...
    -.5*((y1-Lam_11\d)'*Lam_11*(y1-Lam_11\d)+alp'*HiStauH*alp-tmpu'*tmpu);
end