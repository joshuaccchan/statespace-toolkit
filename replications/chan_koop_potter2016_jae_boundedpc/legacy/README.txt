This zip file contains Matlab code for replicating part of the empirical
work in Chan, Koop and Potter (2016). The main files are:

biUC.m - it estimates the bounded bivariate unobserved components model with 
stochastic volatility in Chan, Koop and Potter (2016). This script also estimates variants that
impose no bounds or that assume a constant Phillips curve.

main_forecasting - it performs a recursive forecasting exercise for these bounded bivariate unobserved
components models

The data consist of quarterly CPI inflation rates and (civilian seasonally adjusted) unemployment rates
from 1948Q1 to 2013Q1 (261 obs) obtained from the Federal Reserve Bank of St. Louis economic database. 
The first and second columns of the data file contain respectively the inflation rates and unemployment rates.

This code is free to use for academic purposes only, provided that the paper is cited as:

Chan, J.C.C., Koop, G. and Potter, S.M. (2016). A Bounded Model of Time Variation in Trend Inflation,
NAIRU and the Phillips Curve, Journal of Applied Econometrics, 31(3), 551-565.

This code comes without technical support of any kind. It is expected to reproduce the results reported in
the paper. Under no circumstances will the author be held responsible for any use (or misuse) of this code in
any way.



