# Citing statespace-toolkit

The code here implements methods from the papers below. If you use it in your work, please
cite the paper behind each method you use. The table maps each replication package, library
function and example to that paper, and the BibTeX for every entry follows. A study usually
needs more than one row: one for the model it estimates and one for each sampler inside it.
To credit the software as well, cite the toolkit record at the end.

## What to cite

| Method | In this repository | Cite |
|---|---|---|
| The precision sampler: a state path drawn in one block from its banded precision matrix | `ssm.simulate_states`; ex01-ex06, ex09, ex10 | [Chan and Jeliazkov (2009)](#chan-and-jeliazkov-2009) |
| The integrated likelihood, with the states integrated out | `ssm.intlike`; ex02 | [Chan and Jeliazkov (2009)](#chan-and-jeliazkov-2009) |
| Marginal likelihoods by importance sampling, with the importance density from the cross-entropy method | ex02; `replications/chan_eisenstat2015_er_mlce` | [Chan and Eisenstat (2015)](#chan-and-eisenstat-2015) |
| Missing data and mixed frequencies: the missing values drawn in one block, and draws conditioned on linear restrictions | `ssm.select_obs`, `ssm.restrict`; ex05, ex10 | [Chan, Poon and Zhu (2023)](#chan-poon-and-zhu-2023); [Mariano and Murasawa (2003)](#mariano-and-murasawa-2003) for the quarterly aggregation of monthly values; [Rue and Held (2005)](#rue-and-held-2005) for the update that imposes linear restrictions |
| Stochastic volatility by the auxiliary mixture sampler | `ssm.ksc_rw_h0`, `ssm.ksc_ar1_mean`, `ssm.ksc_rw_diffuse`; ex07, ex08, ex10 | [Kim, Shephard and Chib (1998)](#kim-shephard-and-chib-1998); [Del Negro and Primiceri (2015)](#del-negro-and-primiceri-2015) for the order of the blocks |
| Stochastic volatility in a noncentered parameterization, and tests of whether a model needs its time variation or its stochastic volatility | `ssm.ksc_rw_noncentered`; `replications/chan2018_er_spectest` | [Chan (2018)](#chan-2018) |
| The accept-reject Metropolis-Hastings step for states whose conditional distribution is not Gaussian; stochastic volatility in mean | `ssm.armh`; ex09; `replications/chan2017_jbes_svm` | [Chan (2017)](#chan-2017) |
| Stochastic volatility with MA(1) and Student-t errors | ex08; `replications/chan_hsiao2014_wiley_sv`, `replications/chan2013_joe_masv` | [Chan and Hsiao (2014)](#chan-and-hsiao-2014); [Chan (2013)](#chan-2013) for moving average stochastic volatility |
| Time-varying parameter MIDAS | ex10 | [Chan, Poon and Zhu (2026)](#chan-poon-and-zhu-2026); [Ghysels, Sinko and Valkanov (2007)](#ghysels-sinko-and-valkanov-2007) for MIDAS regressions |
| The output gap from an unobserved components model, and the Hodrick-Prescott filter as one | ex04; `replications/grant_chan2017_jedc_hpfilter` | [Grant and Chan (2017b)](#grant-and-chan-2017b); [Morley, Nelson and Zivot (2003)](#morley-nelson-and-zivot-2003) for the AR(2) cycle; [Ritter and Tanner (1992)](#ritter-and-tanner-1992) for the Griddy-Gibbs step |
| Trend-cycle decompositions of output compared by marginal likelihood | `replications/grant_chan2017_jmcb_trendcycle` | [Grant and Chan (2017a)](#grant-and-chan-2017a) |
| A trend inflation model whose trend stays within bounds | `replications/chan_koop_potter2013_jbes_trendbound` | [Chan, Koop and Potter (2013)](#chan-koop-and-potter-2013) |
| Trend inflation, the NAIRU and the Phillips curve, with bounded time variation | `replications/chan_koop_potter2016_jae_boundedpc` | [Chan, Koop and Potter (2016)](#chan-koop-and-potter-2016) |
| Trend inflation tied to long-run inflation expectations | `replications/chan_clark_koop2018_jmcb_trendie` | [Chan, Clark and Koop (2018)](#chan-clark-and-koop-2018) |
| The uncertainty of inflation expectations, from high-frequency data | `replications/chan_song2018_jmcb_inflrv` | [Chan and Song (2018)](#chan-and-song-2018) |
| The deviance information criterion for latent variable models | `replications/chan_grant2016_csda_dic` | [Chan and Grant (2016a)](#chan-and-grant-2016a) |
| GARCH against stochastic volatility, for energy prices | `replications/chan_grant2016_eneco_garchsv` | [Chan and Grant (2016b)](#chan-and-grant-2016b) |
| The observed-data deviance information criterion for volatility models | `replications/chan_grant2016_jfec_dicsv` | [Chan and Grant (2016c)](#chan-and-grant-2016c) |
| The Kalman filter with backward sampling | ex03 | [Carter and Kohn (1994)](#carter-and-kohn-1994); [Frühwirth-Schnatter (1994)](#frühwirth-schnatter-1994) |
| FRED-MD data | ex06 | [McCracken and Ng (2016)](#mccracken-and-ng-2016) |
| Diagnostics for MCMC output: inefficiency factors, Monte Carlo standard errors and the convergence diagnostic | `ssm.inefficiency_factor`, `ssm.mcse`, `ssm.geweke`, `ssm.specvar0` | [Geweke (1992)](#geweke-1992); [Newey and West (1987)](#newey-and-west-1987) for the long-run variance and [Newey and West (1994)](#newey-and-west-1994) for the default lag of `ssm.geweke` |
| The other library functions: `ssm.diffmat`, `ssm.lagpolymat`, `ssm.surform`, `ssm.mode_newton`, `ssm.tnormrnd` and `ssm.shaded_band` | | [Chan (forthcoming)](#chan-forthcoming) |

## References

### Carter and Kohn (1994)

Carter, C. K. and Kohn, R. (1994). On Gibbs Sampling for State Space Models. *Biometrika*
81(3): 541-553.
[Journal version](https://doi.org/10.1093/biomet/81.3.541)

The Kalman filter with backward sampling, against which ex03 times the precision sampler.

```bibtex
@article{CK94,
  author  = {Carter, C. K. and Kohn, R.},
  title   = {On {G}ibbs Sampling for State Space Models},
  journal = {Biometrika},
  year    = {1994},
  volume  = {81},
  number  = {3},
  pages   = {541--553},
  doi     = {10.1093/biomet/81.3.541}
}
```

### Chan (2013)

Chan, J. C. C. (2013). Moving Average Stochastic Volatility Models with Application to
Inflation Forecast. *Journal of Econometrics* 176(2): 162-172.
[Journal version](https://doi.org/10.1016/j.jeconom.2013.05.003) ·
[Working paper](https://joshuachan.org/papers/MASV.pdf) ·
Code: [`replications/chan2013_joe_masv`](replications/chan2013_joe_masv)

Stochastic volatility with moving average errors, applied to inflation. The SV-MA model of ex08
is a special case.

```bibtex
@article{chan13joe,
  author  = {Chan, J. C. C.},
  title   = {Moving Average Stochastic Volatility Models with Application to Inflation Forecast},
  journal = {Journal of Econometrics},
  year    = {2013},
  volume  = {176},
  number  = {2},
  pages   = {162--172},
  doi     = {10.1016/j.jeconom.2013.05.003}
}
```

### Chan (2017)

Chan, J. C. C. (2017). The Stochastic Volatility in Mean Model with Time-Varying Parameters: An
Application to Inflation Modeling. *Journal of Business and Economic Statistics* 35(1): 17-28.
[Journal version](https://doi.org/10.1080/07350015.2015.1052459) ·
[Working paper](https://joshuachan.org/papers/SVM.pdf) ·
Code: [`replications/chan2017_jbes_svm`](replications/chan2017_jbes_svm)

A stochastic volatility in mean model whose coefficients evolve over time, estimated by an
accept-reject Metropolis-Hastings step that draws the whole log-volatility path in one block from
a Gaussian proposal at the mode of its conditional density. `ssm.armh` implements that step, and
ex09 estimates a simpler version of the model.

```bibtex
@article{chan17jbes,
  author  = {Chan, J. C. C.},
  title   = {The Stochastic Volatility in Mean Model with Time-Varying Parameters: An Application to Inflation Modeling},
  journal = {Journal of Business and Economic Statistics},
  year    = {2017},
  volume  = {35},
  number  = {1},
  pages   = {17--28},
  doi     = {10.1080/07350015.2015.1052459}
}
```

### Chan (2018)

Chan, J. C. C. (2018). Specification Tests for Time-Varying Parameter Models with Stochastic
Volatility. *Econometric Reviews* 37(8): 807-823.
[Journal version](https://doi.org/10.1080/07474938.2016.1167948) ·
[Working paper](https://joshuachan.org/papers/spectest.pdf) ·
Code: [`replications/chan2018_er_spectest`](replications/chan2018_er_spectest)

Tests of time variation in coefficients and volatilities. Under a noncentered parameterization,
the Bayes factor is a Savage-Dickey density ratio, so no marginal likelihood is computed.
`ssm.ksc_rw_noncentered` draws the log-volatility in that parameterization.

```bibtex
@article{chan18er,
  author  = {Chan, J. C. C.},
  title   = {Specification Tests for Time-Varying Parameter Models with Stochastic Volatility},
  journal = {Econometric Reviews},
  year    = {2018},
  volume  = {37},
  number  = {8},
  pages   = {807--823},
  doi     = {10.1080/07474938.2016.1167948}
}
```

### Chan (forthcoming)

Chan, J. C. C. (forthcoming). *Bayesian Macroeconometrics: Methods and Applications*. Chapman &
Hall/CRC.
[Sample chapters](https://joshuachan.org/papers/BayesMacroBook_sample.pdf) ·
Code: [bayesian-macroeconometrics](https://github.com/joshuaccchan/bayesian-macroeconometrics)

The textbook treatment of the models and samplers in this repository, and the reference for the
library functions that no paper in this list introduces.

```bibtex
@book{chan-bmar,
  author    = {Chan, Joshua C. C.},
  title     = {Bayesian Macroeconometrics: Methods and Applications},
  publisher = {Chapman \& Hall/CRC},
  note      = {Forthcoming}
}
```

### Chan, Clark and Koop (2018)

Chan, J. C. C., Clark, T. E. and Koop, G. (2018). A New Model of Inflation, Trend Inflation, and
Long-Run Inflation Expectations. *Journal of Money, Credit and Banking* 50(1): 5-53.
[Journal version](https://doi.org/10.1111/jmcb.12452) ·
[Working paper](https://joshuachan.org/papers/cck1.pdf) ·
Code: [`replications/chan_clark_koop2018_jmcb_trendie`](replications/chan_clark_koop2018_jmcb_trendie)

A model of inflation in which trend inflation is tied to long-run inflation expectations.

```bibtex
@article{CCK18,
  author  = {Chan, J. C. C. and Clark, T. E. and Koop, G.},
  title   = {A New Model of Inflation, Trend Inflation, and Long-Run Inflation Expectations},
  journal = {Journal of Money, Credit and Banking},
  year    = {2018},
  volume  = {50},
  number  = {1},
  pages   = {5--53},
  doi     = {10.1111/jmcb.12452}
}
```

### Chan and Eisenstat (2015)

Chan, J. C. C. and Eisenstat, E. (2015). Marginal Likelihood Estimation with the Cross-Entropy
Method. *Econometric Reviews* 34(3): 256-285.
[Journal version](https://doi.org/10.1080/07474938.2014.944474) ·
[Working paper](https://joshuachan.org/papers/Chan-Eisenstat%202012.pdf) ·
Code: [`replications/chan_eisenstat2015_er_mlce`](replications/chan_eisenstat2015_er_mlce)

Marginal likelihoods estimated by importance sampling, with the importance density chosen by
the cross-entropy method. ex02 builds its importance density from this approach.

```bibtex
@article{CE15,
  author  = {Chan, J. C. C. and Eisenstat, E.},
  title   = {Marginal Likelihood Estimation with the Cross-Entropy Method},
  journal = {Econometric Reviews},
  year    = {2015},
  volume  = {34},
  number  = {3},
  pages   = {256--285},
  doi     = {10.1080/07474938.2014.944474}
}
```

### Chan and Grant (2016a)

Chan, J. C. C. and Grant, A. L. (2016). Fast Computation of the Deviance Information Criterion
for Latent Variable Models. *Computational Statistics and Data Analysis* 100: 847-859.
[Journal version](https://doi.org/10.1016/j.csda.2014.07.018) ·
[Working paper](https://joshuachan.org/papers/Chan-Grant-2014.pdf) ·
Code: [`replications/chan_grant2016_csda_dic`](replications/chan_grant2016_csda_dic)

The deviance information criterion for latent variable models, and how to compute it quickly.

```bibtex
@article{CG16csda,
  author  = {Chan, J. C. C. and Grant, A. L.},
  title   = {Fast Computation of the Deviance Information Criterion for Latent Variable Models},
  journal = {Computational Statistics and Data Analysis},
  year    = {2016},
  volume  = {100},
  pages   = {847--859},
  doi     = {10.1016/j.csda.2014.07.018}
}
```

### Chan and Grant (2016b)

Chan, J. C. C. and Grant, A. L. (2016). Modeling Energy Price Dynamics: GARCH versus Stochastic
Volatility. *Energy Economics* 54: 182-189.
[Journal version](https://doi.org/10.1016/j.eneco.2015.12.003) ·
[Working paper](https://joshuachan.org/papers/energy_GARCH_SV.pdf) ·
Code: [`replications/chan_grant2016_eneco_garchsv`](replications/chan_grant2016_eneco_garchsv)

GARCH and stochastic volatility models of energy prices, compared.

```bibtex
@article{CG16eneco,
  author  = {Chan, J. C. C. and Grant, A. L.},
  title   = {Modeling Energy Price Dynamics: {GARCH} versus Stochastic Volatility},
  journal = {Energy Economics},
  year    = {2016},
  volume  = {54},
  pages   = {182--189},
  doi     = {10.1016/j.eneco.2015.12.003}
}
```

### Chan and Grant (2016c)

Chan, J. C. C. and Grant, A. L. (2016). On the Observed-Data Deviance Information Criterion for
Volatility Modeling. *Journal of Financial Econometrics* 14(4): 772-802.
[Journal version](https://doi.org/10.1093/jjfinec/nbw002) ·
[Working paper](https://joshuachan.org/papers/DIC_SV.pdf) ·
Code: [`replications/chan_grant2016_jfec_dicsv`](replications/chan_grant2016_jfec_dicsv)

The observed-data deviance information criterion for volatility models.

```bibtex
@article{CG16jfec,
  author  = {Chan, J. C. C. and Grant, A. L.},
  title   = {On the Observed-Data Deviance Information Criterion for Volatility Modeling},
  journal = {Journal of Financial Econometrics},
  year    = {2016},
  volume  = {14},
  number  = {4},
  pages   = {772--802},
  doi     = {10.1093/jjfinec/nbw002}
}
```

### Chan and Hsiao (2014)

Chan, J. C. C. and Hsiao, C. Y. L. (2014). Estimation of Stochastic Volatility Models with Heavy
Tails and Serial Dependence. In I. Jeliazkov and X.-S. Yang (Eds.), *Bayesian Inference in the
Social Sciences*, 155-176. John Wiley & Sons, Hoboken, New Jersey.
[Chapter](https://doi.org/10.1002/9781118771051.ch6) ·
[Working paper](https://joshuachan.org/papers/Chan-Hsiao-2013.pdf) ·
Code: [`replications/chan_hsiao2014_wiley_sv`](replications/chan_hsiao2014_wiley_sv)

The standard stochastic volatility model and its variants with MA(1) Gaussian and MA(1)
Student-t errors, which ex08 estimates with mean zero.

```bibtex
@incollection{CH14,
  author    = {Chan, J. C. C. and Hsiao, C. Y. L.},
  title     = {Estimation of Stochastic Volatility Models with Heavy Tails and Serial Dependence},
  booktitle = {Bayesian Inference in the Social Sciences},
  editor    = {Jeliazkov, I. and Yang, X.-S.},
  publisher = {John Wiley \& Sons},
  address   = {Hoboken, NJ},
  year      = {2014},
  pages     = {155--176},
  doi       = {10.1002/9781118771051.ch6}
}
```

### Chan and Jeliazkov (2009)

Chan, J. C. C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated Likelihood
Estimation in State Space Models. *International Journal of Mathematical Modelling and
Numerical Optimisation* 1(1/2): 101-120.
[Journal version](https://doi.org/10.1504/IJMMNO.2009.030090) ·
[Working paper](https://joshuachan.org/papers/statespace1.pdf) ·
Code: [bvar-toolkit, `replications/chan_jeliazkov2009_statespace`](https://github.com/joshuaccchan/bvar-toolkit/tree/main/replications/chan_jeliazkov2009_statespace)

A derivation of the posterior distribution of the states that leads to a modular, scalable
precision-based simulation algorithm for state space models, and a simple way to evaluate the
integrated likelihood. `ssm.simulate_states` and `ssm.intlike` implement them.

```bibtex
@article{CJ09,
  author  = {Chan, J. C. C. and Jeliazkov, I.},
  title   = {Efficient Simulation and Integrated Likelihood Estimation in State Space Models},
  journal = {International Journal of Mathematical Modelling and Numerical Optimisation},
  year    = {2009},
  volume  = {1},
  number  = {1/2},
  pages   = {101--120},
  doi     = {10.1504/IJMMNO.2009.030090}
}
```

### Chan, Koop and Potter (2013)

Chan, J. C. C., Koop, G. and Potter, S. M. (2013). A New Model of Trend Inflation. *Journal of
Business and Economic Statistics* 31(1): 94-106.
[Journal version](https://doi.org/10.1080/07350015.2012.741549) ·
[Working paper](https://joshuachan.org/papers/ckp1.pdf) ·
Code: [`replications/chan_koop_potter2013_jbes_trendbound`](replications/chan_koop_potter2013_jbes_trendbound)

A trend inflation model whose trend stays within bounds.

```bibtex
@article{CKP13,
  author  = {Chan, J. C. C. and Koop, G. and Potter, S. M.},
  title   = {A New Model of Trend Inflation},
  journal = {Journal of Business and Economic Statistics},
  year    = {2013},
  volume  = {31},
  number  = {1},
  pages   = {94--106},
  doi     = {10.1080/07350015.2012.741549}
}
```

### Chan, Koop and Potter (2016)

Chan, J. C. C., Koop, G. and Potter, S. M. (2016). A Bounded Model of Time Variation in Trend
Inflation, NAIRU and the Phillips Curve. *Journal of Applied Econometrics* 31(3): 551-565.
[Journal version](https://doi.org/10.1002/jae.2442) ·
[Working paper](https://joshuachan.org/papers/ckp2.pdf) ·
Code: [`replications/chan_koop_potter2016_jae_boundedpc`](replications/chan_koop_potter2016_jae_boundedpc)

Trend inflation, the NAIRU and the Phillips curve, with bounded time variation.

```bibtex
@article{CKP16,
  author  = {Chan, J. C. C. and Koop, G. and Potter, S. M.},
  title   = {A Bounded Model of Time Variation in Trend Inflation, {NAIRU} and the {P}hillips Curve},
  journal = {Journal of Applied Econometrics},
  year    = {2016},
  volume  = {31},
  number  = {3},
  pages   = {551--565},
  doi     = {10.1002/jae.2442}
}
```

### Chan, Poon and Zhu (2023)

Chan, J. C. C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian State
Space Models with Missing Data. *Journal of Econometrics* 236(1): 105468.
[Journal version](https://doi.org/10.1016/j.jeconom.2023.05.005) ·
[Working paper](https://joshuachan.org/papers/BVAR-MF-R1.pdf)

An efficient approach to sampling the missing values of a conditionally Gaussian state space
model in one block, from a conditional distribution whose precision matrix is banded for common
missing data patterns, with restrictions that tie missing high-frequency values to observed
low-frequency ones. `ssm.select_obs` and ex05 implement the draw.

```bibtex
@article{CPZ23,
  author  = {Chan, J. C. C. and Poon, A. and Zhu, D.},
  title   = {High-Dimensional Conditionally {G}aussian State Space Models with Missing Data},
  journal = {Journal of Econometrics},
  year    = {2023},
  volume  = {236},
  number  = {1},
  pages   = {105468},
  doi     = {10.1016/j.jeconom.2023.05.005}
}
```

### Chan, Poon and Zhu (2026)

Chan, J. C. C., Poon, A. and Zhu, D. (2026). Time-Varying Parameter MIDAS Models: Application to
Nowcasting US Real GDP. *Journal of Econometrics*, forthcoming.
[Journal version](https://doi.org/10.1016/j.jeconom.2025.106090) ·
[Working paper](https://joshuachan.org/papers/TVP_MIDAS.pdf)

MIDAS regressions whose weights and coefficients vary over time, with a weighting function that
is linear in its parameters, applied to nowcasting US real GDP. ex10 estimates the model on
generated data.

```bibtex
@article{CPZ26,
  author  = {Chan, J. C. C. and Poon, A. and Zhu, D.},
  title   = {Time-Varying Parameter {MIDAS} Models: Application to Nowcasting {US} Real {GDP}},
  journal = {Journal of Econometrics},
  year    = {2026},
  note    = {Forthcoming},
  doi     = {10.1016/j.jeconom.2025.106090}
}
```

### Chan and Song (2018)

Chan, J. C. C. and Song, Y. (2018). Measuring Inflation Expectations Uncertainty Using
High-Frequency Data. *Journal of Money, Credit and Banking* 50(6): 1139-1166.
[Journal version](https://doi.org/10.1111/jmcb.12498) ·
[Working paper](https://joshuachan.org/papers/inflation-RV.pdf) ·
Code: [`replications/chan_song2018_jmcb_inflrv`](replications/chan_song2018_jmcb_inflrv)

The uncertainty of inflation expectations, measured from high-frequency data.

```bibtex
@article{CS18,
  author  = {Chan, J. C. C. and Song, Y.},
  title   = {Measuring Inflation Expectations Uncertainty Using High-Frequency Data},
  journal = {Journal of Money, Credit and Banking},
  year    = {2018},
  volume  = {50},
  number  = {6},
  pages   = {1139--1166},
  doi     = {10.1111/jmcb.12498}
}
```

### Del Negro and Primiceri (2015)

Del Negro, M. and Primiceri, G. E. (2015). Time Varying Structural Vector Autoregressions and
Monetary Policy: A Corrigendum. *Review of Economic Studies* 82(4): 1342-1345.
[Journal version](https://doi.org/10.1093/restud/rdv024)

The order of the blocks in a sampler that uses the auxiliary mixture sampler: the mixture
indicators are drawn immediately before the log-volatility. ex07, ex08 and ex10 follow it.

```bibtex
@article{DP15,
  author  = {Del Negro, M. and Primiceri, G. E.},
  title   = {Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum},
  journal = {Review of Economic Studies},
  year    = {2015},
  volume  = {82},
  number  = {4},
  pages   = {1342--1345},
  doi     = {10.1093/restud/rdv024}
}
```

### Frühwirth-Schnatter (1994)

Frühwirth-Schnatter, S. (1994). Data Augmentation and Dynamic Linear Models. *Journal of Time
Series Analysis* 15(2): 183-202.
[Journal version](https://doi.org/10.1111/j.1467-9892.1994.tb00184.x)

The Kalman filter with backward sampling, derived independently of Carter and Kohn (1994).

```bibtex
@article{FS94,
  author  = {Fr{\"u}hwirth-Schnatter, S.},
  title   = {Data Augmentation and Dynamic Linear Models},
  journal = {Journal of Time Series Analysis},
  year    = {1994},
  volume  = {15},
  number  = {2},
  pages   = {183--202},
  doi     = {10.1111/j.1467-9892.1994.tb00184.x}
}
```

### Geweke (1992)

Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of
Posterior Moments. In J. M. Bernardo, J. O. Berger, A. P. Dawid and A. F. M. Smith (Eds),
*Bayesian Statistics 4*, 169-193. Oxford University Press.
[Publisher version](https://doi.org/10.1093/oso/9780198522669.003.0010) ·
[Working paper](https://doi.org/10.21034/sr.148)

Numerical standard errors and relative numerical efficiency from the spectral density at
frequency zero, and the convergence diagnostic that compares the means of an early and a late
segment of a chain.

```bibtex
@incollection{Geweke92,
  author    = {Geweke, J.},
  title     = {Evaluating the Accuracy of Sampling-Based Approaches to the Calculation of Posterior Moments},
  booktitle = {{B}ayesian Statistics 4},
  editor    = {Bernardo, J. M. and Berger, J. O. and Dawid, A. P. and Smith, A. F. M.},
  publisher = {Oxford University Press},
  year      = {1992},
  pages     = {169--193},
  doi       = {10.1093/oso/9780198522669.003.0010}
}
```

### Ghysels, Sinko and Valkanov (2007)

Ghysels, E., Sinko, A. and Valkanov, R. (2007). MIDAS Regressions: Further Results and New
Directions. *Econometric Reviews* 26(1): 53-90.
[Journal version](https://doi.org/10.1080/07474930600972467)

MIDAS regressions and their weighting functions, on which the model of ex10 builds.

```bibtex
@article{GSV07,
  author  = {Ghysels, E. and Sinko, A. and Valkanov, R.},
  title   = {{MIDAS} Regressions: Further Results and New Directions},
  journal = {Econometric Reviews},
  year    = {2007},
  volume  = {26},
  number  = {1},
  pages   = {53--90},
  doi     = {10.1080/07474930600972467}
}
```

### Grant and Chan (2017a)

Grant, A. L. and Chan, J. C. C. (2017). A Bayesian Model Comparison for Trend-Cycle
Decompositions of Output. *Journal of Money, Credit and Banking* 49(2-3): 525-552.
[Journal version](https://doi.org/10.1111/jmcb.12388) ·
[Working paper](https://joshuachan.org/papers/output-gap.pdf) ·
Code: [`replications/grant_chan2017_jmcb_trendcycle`](replications/grant_chan2017_jmcb_trendcycle)

Trend-cycle decompositions of output, compared by marginal likelihood.

```bibtex
@article{GC17jmcb,
  author  = {Grant, A. L. and Chan, J. C. C.},
  title   = {A {B}ayesian Model Comparison for Trend-Cycle Decompositions of Output},
  journal = {Journal of Money, Credit and Banking},
  year    = {2017},
  volume  = {49},
  number  = {2-3},
  pages   = {525--552},
  doi     = {10.1111/jmcb.12388}
}
```

### Grant and Chan (2017b)

Grant, A. L. and Chan, J. C. C. (2017). Reconciling Output Gaps: Unobserved Components Model
and Hodrick-Prescott Filter. *Journal of Economic Dynamics and Control* 75: 114-121.
[Journal version](https://doi.org/10.1016/j.jedc.2016.12.004) ·
[Working paper](https://joshuachan.org/papers/output-gap-2M.pdf) ·
Code: [`replications/grant_chan2017_jedc_hpfilter`](replications/grant_chan2017_jedc_hpfilter)

The Hodrick-Prescott filter as an unobserved components model. The model of ex04 is similar to
the paper's.

```bibtex
@article{GC17jedc,
  author  = {Grant, A. L. and Chan, J. C. C.},
  title   = {Reconciling Output Gaps: Unobserved Components Model and {H}odrick-{P}rescott Filter},
  journal = {Journal of Economic Dynamics and Control},
  year    = {2017},
  volume  = {75},
  pages   = {114--121},
  doi     = {10.1016/j.jedc.2016.12.004}
}
```

### Kim, Shephard and Chib (1998)

Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference and
Comparison with ARCH Models. *Review of Economic Studies* 65(3): 361-393.
[Journal version](https://doi.org/10.1111/1467-937X.00050)

The seven-component normal mixture approximation that the auxiliary mixture samplers
`ssm.ksc_rw_h0`, `ssm.ksc_ar1_mean` and `ssm.ksc_rw_diffuse` use, and the reweighting of the
draws to the exact posterior that ex07 illustrates.

```bibtex
@article{KSC98,
  author  = {Kim, S. and Shephard, N. and Chib, S.},
  title   = {Stochastic Volatility: Likelihood Inference and Comparison with {ARCH} Models},
  journal = {Review of Economic Studies},
  year    = {1998},
  volume  = {65},
  number  = {3},
  pages   = {361--393},
  doi     = {10.1111/1467-937X.00050}
}
```

### Mariano and Murasawa (2003)

Mariano, R. S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based on
Monthly and Quarterly Series. *Journal of Applied Econometrics* 18(4): 427-443.
[Journal version](https://doi.org/10.1002/jae.695)

A coincident index of business cycles from monthly and quarterly series, with the log-linear
approximation that ties a quarterly growth rate to the monthly growth rates of the same series.
ex05 uses the approximation.

```bibtex
@article{MM03,
  author  = {Mariano, R. S. and Murasawa, Y.},
  title   = {A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series},
  journal = {Journal of Applied Econometrics},
  year    = {2003},
  volume  = {18},
  number  = {4},
  pages   = {427--443},
  doi     = {10.1002/jae.695}
}
```

### McCracken and Ng (2016)

McCracken, M. W. and Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research.
*Journal of Business and Economic Statistics* 34(4): 574-589.
[Journal version](https://doi.org/10.1080/07350015.2015.1086655)

The monthly database that ex06 draws its data from.

```bibtex
@article{MN16,
  author  = {McCracken, M. W. and Ng, S.},
  title   = {{FRED-MD}: A Monthly Database for Macroeconomic Research},
  journal = {Journal of Business and Economic Statistics},
  year    = {2016},
  volume  = {34},
  number  = {4},
  pages   = {574--589},
  doi     = {10.1080/07350015.2015.1086655}
}
```

### Morley, Nelson and Zivot (2003)

Morley, J. C., Nelson, C. R. and Zivot, E. (2003). Why Are the Beveridge-Nelson and
Unobserved-Components Decompositions of GDP So Different? *Review of Economics and Statistics*
85(2): 235-243.
[Journal version](https://doi.org/10.1162/003465303765299765)

The unobserved components model of output with an AR(2) cycle, the cycle that ex04 uses.

```bibtex
@article{MNZ03,
  author  = {Morley, J. C. and Nelson, C. R. and Zivot, E.},
  title   = {Why Are the {B}everidge-{N}elson and Unobserved-Components Decompositions of {GDP} So Different?},
  journal = {Review of Economics and Statistics},
  year    = {2003},
  volume  = {85},
  number  = {2},
  pages   = {235--243},
  doi     = {10.1162/003465303765299765}
}
```

### Newey and West (1987)

Newey, W. K. and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and
Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3): 703-708.
[Journal version](https://doi.org/10.2307/1913610)

The Bartlett-window estimator of the long-run variance that `ssm.specvar0` computes.

```bibtex
@article{NW87,
  author  = {Newey, W. K. and West, K. D.},
  title   = {A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  year    = {1987},
  volume  = {55},
  number  = {3},
  pages   = {703--708},
  doi     = {10.2307/1913610}
}
```

### Newey and West (1994)

Newey, W. K. and West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
*Review of Economic Studies* 61(4): 631-653.
[Journal version](https://doi.org/10.2307/2297912)

The rule of thumb for the truncation lag that `ssm.geweke` uses by default.

```bibtex
@article{NW94,
  author  = {Newey, W. K. and West, K. D.},
  title   = {Automatic Lag Selection in Covariance Matrix Estimation},
  journal = {Review of Economic Studies},
  year    = {1994},
  volume  = {61},
  number  = {4},
  pages   = {631--653},
  doi     = {10.2307/2297912}
}
```

### Ritter and Tanner (1992)

Ritter, C. and Tanner, M. A. (1992). Facilitating the Gibbs Sampler: The Gibbs Stopper and the
Griddy-Gibbs Sampler. *Journal of the American Statistical Association* 87(419): 861-868.
[Journal version](https://doi.org/10.1080/01621459.1992.10475289)

The Griddy-Gibbs sampler, which ex04 uses to draw the variance of trend growth.

```bibtex
@article{RT92,
  author  = {Ritter, C. and Tanner, M. A.},
  title   = {Facilitating the {G}ibbs Sampler: The {G}ibbs Stopper and the {G}riddy-{G}ibbs Sampler},
  journal = {Journal of the American Statistical Association},
  year    = {1992},
  volume  = {87},
  number  = {419},
  pages   = {861--868},
  doi     = {10.1080/01621459.1992.10475289}
}
```

### Rue and Held (2005)

Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman &
Hall/CRC, Boca Raton.
[Book](https://doi.org/10.1201/9780203492024)

Algorithm 2.6 conditions a Gaussian draw on linear equality restrictions, the update of
`ssm.restrict`, which ex05 and ex10 use.

```bibtex
@book{RH05,
  author    = {Rue, H. and Held, L.},
  title     = {{G}aussian {M}arkov Random Fields: Theory and Applications},
  publisher = {Chapman \& Hall/CRC},
  address   = {Boca Raton},
  year      = {2005},
  doi       = {10.1201/9780203492024}
}
```

## The toolkit

Chan, J. C. C. (2026). *statespace-toolkit: MATLAB code for Bayesian state space models*.
Zenodo. https://doi.org/10.5281/zenodo.22884655

The DOI resolves to the latest release. `CITATION.cff` holds the same record in machine-readable
form, which GitHub's "Cite this repository" button reads.

```bibtex
@misc{chan26statespacetoolkit,
  author       = {Chan, J. C. C.},
  title        = {statespace-toolkit: {MATLAB} Code for {B}ayesian State Space Models},
  year         = {2026},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.22884655},
  url          = {https://github.com/joshuaccchan/statespace-toolkit}
}
```
