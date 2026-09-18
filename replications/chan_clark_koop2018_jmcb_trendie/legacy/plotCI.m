% Support function to estimate the trend inflation models in Chan, Clark 
% and Koop (2018)
% 
% See:
% Chan, J.C.C., T. E. Clark, and G. Koop (2018). A New Model of Inflation,
% Trend Inflation, and Long-Run Inflation Expectations, Journal of Money, 
% Credit and Banking, 50(1), 5-53.

function plotCI(x,y1,y2,C)
if nargin == 3
    C = [.85 .85 .85];
end    
f = fill([x',fliplr(x')], [y1',fliplr(y2')],C);
set(f,'EdgeColor','none'),
end