This zip file contains Matlab code for replicating the three empirical examples in 
Chan and Grant (2015).  The main files are:

1. SF.m: this script estimates a static factor model and computes the 
   observed-data DIC (DIC2), complete-data DIC (DIC5) and conditional DIC (DIC7).

2. VAR1.m: this script estimates a VAR(1) and computes the (observed-data) DIC.

3. TVPVAR.m: this script estimates a TVP-VAR and computes the observed-data
   DIC (DIC2), complete-data DIC (DIC5) and conditional DIC (DIC7).

4. CTVPVAR.m: this script estimates a variant of the TVP-VAR where the first
   equation has constant coefficients. This file also computes the observed-data
   DIC (DIC2), complete-data DIC (DIC5) and conditional DIC (DIC7) under the model.

5. semireg.m: this script estimates a semiparametric model and computes the 
   observed-data DIC (DIC2), complete-data DIC (DIC5) and conditional DIC (DIC7).

Note that the BMI data are restricted access and so they cannot be uploaded
directly. The data we use in this paper, however, are the same
as the data that were used in the paper, “The Wages of BMI: Bayesian 
Analysis of a Skewed Treatment-Response Model with Nonparametric 
Endogeneity” (Kline and Tobias, 2008).


This code is free to use for academic purposes only, provided that the 
paper is cited as:

Chan, J. C. C. and Grant, A. L. (2016). "Fast Computation of the Deviance
Information Criterion for Latent Variable Models," Computational 
Statistics and Data Analysis, 100, 847-859.

This code comes without technical support of any kind.  It
is expected to reproduce the results reported in the paper.
Under no circumstances will the authors be held responsible for any use
(or misuse) of this code in any way.
