% ssm.select_obs - the selection matrices of a missing data pattern, y = So*yo + Sm*ym,
% splitting the stacked data into its observed and missing values.
%
%   [So, Sm] = ssm.select_obs(Y)
%   [So, Sm, yo] = ssm.select_obs(Y)
%
%   Y  : T x n data matrix, NaN wherever the value is missing
%   So : Tn x No sparse selection matrix for the observed values
%   Sm : Tn x Nm sparse selection matrix for the missing values
%   yo : No x 1 vector of observed values, ordered as So takes them
%
% y = (y_1', ..., y_T')' stacks the n variables within each period, and yo and ym follow
% that order, so the precision of (ym | yo), Gm'*inv(Sigma)*Gm, keeps the bandedness of
% Sigma and of the difference matrices in Gm. [So, Sm] is a permutation matrix, so both
% factors have full column rank and Sm'*y recovers the missing values of a completed y.
% An all-observed or all-missing pattern gives the corresponding matrix Tn x 0. Written
% for this toolkit.
%
% See:
% Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
% State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
% Section 2.1.

function [So, Sm, yo] = select_obs(Y)
if ~isnumeric(Y) || ~ismatrix(Y) || isempty(Y)
    error('ssm:select_obs:badY', 'Y must be a nonempty T x n numeric matrix');
end
[T, n] = size(Y);
y = reshape(Y', T*n, 1);                       % y = (y_1', ..., y_T')'
io = find(~isnan(y));
im = find(isnan(y));
So = sparse(io, (1:numel(io))', ones(numel(io),1), T*n, numel(io));
Sm = sparse(im, (1:numel(im))', ones(numel(im),1), T*n, numel(im));
yo = y(io);
end
