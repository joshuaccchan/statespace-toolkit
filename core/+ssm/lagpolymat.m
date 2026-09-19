% ssm.lagpolymat - the sparse matrix of the lag polynomial 1 - phi_1*L - ... - phi_p*L^p
% applied to a stacked path, where L is the lag operator.
%
%   H = ssm.lagpolymat(T, phi)
%
%   T   : path length, a positive integer
%   phi : vector of lag coefficients (phi_1, ..., phi_p); empty gives the identity
%   H   : T x T sparse lower-banded matrix, 1 on the diagonal and -phi_j on the jth
%         subdiagonal
%
% For x_t = phi_1*x_{t-1} + ... + phi_p*x_{t-p} + u_t with the pre-sample values set to
% zero, H*x = u; nonzero pre-sample values enter through a vector the caller adds.
% Second differences are phi = [2 -1], an AR(2) is phi = [phi_1 phi_2], and for a
% scalar a, ssm.lagpolymat(T, a) equals ssm.diffmat(T, a). Written for this toolkit.
%
% See:
% Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and
% Applications, Chapman & Hall/CRC, Section 9.1.3.

function H = lagpolymat(T, phi)
if ~isscalar(T) || T < 1 || T ~= round(T)
    error('ssm:lagpolymat:badT', 'T must be a positive integer');
end
if ~isnumeric(phi) || ~(isvector(phi) || isempty(phi))
    error('ssm:lagpolymat:badPhi', 'phi must be a vector of lag coefficients');
end
H = speye(T);
for j = 1:min(numel(phi), T-1)
    H = H - phi(j)*sparse(j+1:T, 1:T-j, 1, T, T);
end
end
