# statespace-toolkit <img src=".github/logo.png" align="right" height="139" alt="statespace-toolkit logo" />

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22884655.svg)](https://doi.org/10.5281/zenodo.22884655) [![View on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/184886-statespace-toolkit-bayesian-state-space-models)

A MATLAB library for Bayesian state space models: estimate latent trends, track time-varying
relationships, model stochastic volatility, and handle missing or mixed-frequency data. The library
builds on the precision sampler of Chan and Jeliazkov (2009), which draws the entire state path
jointly, with reusable routines for integrated likelihood evaluation and stochastic volatility
sampling. Worked [examples](examples/) cover unobserved components, time-varying parameter,
stochastic volatility and dynamic factor models.

A Python version,
[statespace-toolkit-python](https://github.com/joshuaccchan/statespace-toolkit-python), has the
same functions under the same names, each tested against its MATLAB twin, and the ten how-to
guides with Python code. The five functions of `ssm.models` have no Python version yet.

## Installation

Download the repository as a ZIP file (the Code button, then Download ZIP) and unzip it, or clone
it with git. The latest release is also on
[File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/184886-statespace-toolkit-bayesian-state-space-models).
In MATLAB, go to the folder that contains `setup.m` and run it:

```matlab
run setup.m                 % adds core/ to the path
```

The code needs MATLAB with the Statistics and Machine Learning Toolbox. `setup.m` also checks for
the Optimization Toolbox, which four of the archived packages use. The library and the examples
are tested on MATLAB R2025b, on Windows and, in the repository's continuous integration, on Linux.

## A First Example

The trend of a local level model, drawn given its parameters:

```matlab
T = 200; sig2 = 1; omega2 = .05;
rng(1)
y = cumsum(sqrt(omega2)*randn(T, 1)) + sqrt(sig2)*randn(T, 1);
H = ssm.diffmat(T);                          % first differences
K = H'*H/omega2 + speye(T)/sig2;             % the precision of the trend given y
[draws, tauhat] = ssm.simulate_states(K, y/sig2, 1000);
```

[How to Draw the States of an Unobserved Components Model](guides/uc_states/) explains each line.
The script [`ex01_precision_sampler`](examples/ex01_precision_sampler.m) in `examples/` checks the
sampler against a Kalman smoother and then estimates the trend of US CPI inflation.

## Learn the Methods

Most of the methods are developed in the book *Bayesian Macroeconometrics: Methods and
Applications* (Chapman & Hall/CRC, 2027): see the
[sample chapters](https://joshuachan.org/papers/BayesMacroBook_sample.pdf) and
[its code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics), with MATLAB,
R and Python for all fourteen chapters. The missing-data draw of ex05 comes from Chan, Poon and
Zhu (2023), and the TVP-MIDAS model of ex10 from Chan, Poon and Zhu (2026).

The [how-to guides](guides/) show how to use the core functions, one step at a time, on small
examples with generated data. The first series draws the states of the local level model with the
precision sampler, estimates its parameters by Gibbs sampling, and integrates the states out of its
likelihood:

- [How to Draw the States of an Unobserved Components Model](guides/uc_states/)
- [How to Estimate a State Space Model by Gibbs Sampling](guides/gibbs_sampler/)
- [How to Compute the Integrated Likelihood of a State Space Model](guides/integrated_likelihood/)

The second series does the same for a regression whose coefficients follow random walks, and then
asks whether a coefficient varies at all:

- [How to Draw Time-Varying Regression Coefficients](guides/tvp_states/)
- [How to Estimate a TVP Regression by Gibbs Sampling](guides/tvp_gibbs/)
- [How to Tell Whether a Coefficient Varies over Time](guides/tvp_compare/)

The third series does the same for a stochastic volatility model, then adds Student-t errors and
compares the models by their marginal likelihoods:

- [How to Draw the Log-Volatilities of a Stochastic Volatility Model](guides/sv_states/)
- [How to Estimate a Stochastic Volatility Model by Gibbs Sampling](guides/sv_gibbs/)
- [How to Add Student-t Errors to a Stochastic Volatility Model](guides/sv_t/)
- [How to Compare Stochastic Volatility Models by the Marginal Likelihood](guides/sv_compare/)

The ten scripts in [`examples/`](examples/) each run in under a minute. Start with ex01.
ex01-ex06 are linear Gaussian models. ex07-ex09 add stochastic volatility: ex07 and ex08 draw the
log-volatility by the auxiliary mixture sampler, and ex09, in which the volatility also enters the
mean, by accept-reject Metropolis-Hastings. ex10 combines the two groups: it imposes the linear
restriction on its time-varying weights with `ssm.restrict`, which ex05 uses for the quarterly
aggregation, and draws its volatility as ex07 does.
[`examples/README.md`](examples/README.md) lists what each one shows and the data it uses.

The [tutorials](tutorials/) answer one empirical question each with published models, on the
data in [`examples/data/`](examples/data/), and can be read without MATLAB. Each comes with
a script that runs the same analysis on your own series.

- [Which Unobserved Components Model Should I Use for Inflation?](tutorials/uc_specification/)
  One with stochastic volatility in the transitory component: such models beat an AR(4) in
  forecasting average US CPI inflation over the next year, and stochastic volatility in the trend
  as well does not help.
- [Which Parameters of a GDP Nowcasting Model Should Vary over Time?](tutorials/midas_nowcast/)
  The coefficients of a mixed-frequency (MIDAS) regression: letting them vary improves the
  nowcasts of US real GDP growth, and letting the weights vary adds little.
- [Which Stochastic Volatility Model Forecasts Best?](tutorials/sv_volatility/)
  One with Student-t errors: it forecasts the daily return on the dollar-euro rate better than the
  RiskMetrics EWMA and than SV with normal errors, and a random-walk log-volatility or an MA(1)
  term does not help.

## Reproduce a Paper

Each package is kept byte for byte as published under `replications/<paper>/legacy/`, with a
permanent `as-published/<paper>` git tag and the source zip's md5 recorded in
[`provenance.md`](provenance.md). Run the entry scripts from their own folders.

### Which Model Do I Want?

| If you want | Paper | Folder | Entry scripts |
|---|---|---|---|
| A trend inflation model whose trend stays within bounds | Chan, Koop & Potter (2013, JBES) | `chan_koop_potter2013_jbes_trendbound` | `ARtrend_bound.m` |
| Stochastic volatility with moving average errors, for inflation | Chan (2013, JoE) | `chan2013_joe_masv` | `main_UCMA.m` |
| Trend inflation, the NAIRU and the Phillips curve, with bounded time variation | Chan, Koop & Potter (2016, JAE) | `chan_koop_potter2016_jae_boundedpc` | `biUC.m`, `main_forecasting.m` |
| Stochastic volatility in mean with time-varying parameters | Chan (2017, JBES) | `chan2017_jbes_svm` | `UC_SVM.m` |
| Trend inflation tied to long-run inflation expectations | Chan, Clark & Koop (2018, JMCB) | `chan_clark_koop2018_jmcb_trendie` | `main_estimation.m`, `main_forecasting.m` |
| The uncertainty of inflation expectations, from high-frequency data | Chan & Song (2018, JMCB) | `chan_song2018_jmcb_inflrv` | `main_inflation_RV.m` |
| Trend-cycle decompositions of output compared by marginal likelihood | Grant & Chan (2017, JMCB) | `grant_chan2017_jmcb_trendcycle` | `main_UC.m` |
| The Hodrick-Prescott filter as an unobserved components model | Grant & Chan (2017, JEDC) | `grant_chan2017_jedc_hpfilter` | `main_script.m` |
| Stochastic volatility with heavy tails and serial dependence | Chan & Hsiao (2014, Wiley) | `chan_hsiao2014_wiley_sv` | `plainSV.m`, `MASV.m`, `MASVt.m` |
| GARCH against stochastic volatility, for energy prices | Chan & Grant (2016, Energy Economics) | `chan_grant2016_eneco_garchsv` | `main_SV.m`, `main_GARCH.m`, `main_forecasting.m` |
| Marginal likelihoods by the cross-entropy method | Chan & Eisenstat (2015, ER) | `chan_eisenstat2015_er_mlce` | `probit_mc.m`, `logit_mc.m`, `bin_t_mc.m`, `VAR_mc.m`, `TVPVAR_mc.m`, `DFVAR_mc.m` |
| The deviance information criterion for latent variable models | Chan & Grant (2016, CSDA) | `chan_grant2016_csda_dic` | `SF.m`, `VAR1.m`, `TVPVAR.m`, `CTVPVAR.m`, `semireg.m` |
| The observed-data deviance information criterion for volatility models | Chan & Grant (2016, JFEC) | `chan_grant2016_jfec_dicsv` | `main_SV.m` |
| Tests of whether a model needs its time variation or its stochastic volatility | Chan (2018, ER) | `chan2018_er_spectest` | `main_NAIRU.m`, `main_UCSV.m` |

[`tests/golden_runs/manifest.md`](tests/golden_runs/manifest.md) names the few legacy scripts that
do not run as shipped.

## Build on the Code

Across the fourteen packages the same code recurs: the truncated normal sampler is copied into six
of them, and the function that builds the design matrix of a time-varying parameter regression
into four. The `ssm` package under `core/` holds one copy of each step. Call the functions
directly, or copy the example closest to your model; every example but ex05 builds a complete
sampler from them.

| Function | Does | Shown in |
|---|---|---|
| `ssm.simulate_states` | Draws the whole state path at once: the precision sampler of Chan and Jeliazkov (2009) | ex01-ex06, ex09, ex10, the UC and MIDAS tutorials |
| `ssm.intlike` | Computes the log likelihood of a linear Gaussian model, with the states integrated out | ex02, the UC tutorial |
| `ssm.intlike_sv` | Computes the log likelihood of an SV model by importance sampling | the guide on SV model comparison |
| `ssm.intlike_sv_t` | The same with Student-t errors | the guide on SV model comparison |
| `ssm.select_obs` | Splits the data into observed and missing values, keeping the precision banded | ex05 |
| `ssm.restrict` | Conditions Gaussian draws on linear restrictions M*x = z | ex05, ex10, the MIDAS tutorial |
| `ssm.diffmat` | Builds the first-difference matrix I - aL | ex01-ex03, ex06, ex07, ex09, ex10, the UC and MIDAS tutorials |
| `ssm.lagpolymat` | Builds the matrix of a lag polynomial, such as an AR(2) | ex04 |
| `ssm.surform` | Builds the design matrix of a regression with time-varying coefficients | ex03, ex09, ex10, the UC and MIDAS tutorials |
| `ssm.midas_basis` | Builds the regressors and restrictions of a MIDAS predictor with Fourier weights | the MIDAS tutorial |
| `ssm.mode_newton` | Finds the mode of a concave log density by Newton-Raphson | ex08, and through `ssm.armh` |
| `ssm.armh` | Draws non-Gaussian states by the accept-reject Metropolis-Hastings step of Chan (2017) | ex09 |
| `ssm.ksc_rw_h0` | Draws a random-walk log-volatility by the auxiliary mixture sampler of Kim, Shephard and Chib (1998) | ex07, ex10, the UC tutorial |
| `ssm.ksc_ar1_mean` | The same for a stationary AR(1) log-volatility | ex08, the UC and volatility tutorials |
| `ssm.ksc_rw_diffuse` | The same for a random walk whose first value has a mean-zero normal prior | all three tutorials |
| `ssm.ksc_rw_noncentered` | Draws a random-walk log-volatility in the noncentered form of Chan (2018) | not yet in an example |
| `ssm.sv_params` | Draws the parameters of AR(1) log-volatilities, given their paths | the guides on SV by Gibbs sampling, Student-t errors and SV model comparison, the volatility tutorial |
| `ssm.nu_studentt` | Draws the degrees of freedom of Student-t errors | the guides on Student-t errors and SV model comparison, the volatility tutorial |
| `ssm.ma_params` | Draws the coefficient of MA(1) errors with time-varying variances | through `ssm.models.uc` and `ssm.models.sv`; the volatility tutorial |
| `ssm.arma_params` | Draws the coefficients of ARMA(1,1) errors with time-varying variances | through `ssm.models.sv` |
| `ssm.tnormrnd` | Draws from a truncated normal distribution | ex02, the UC tutorial |
| `ssm.shaded_band` | Shades credible bands in figures | ex01, ex03, ex04, ex07-ex09, the MIDAS tutorial |
| `ssm.inefficiency_factor` | Computes the inefficiency factor of MCMC draws | the Gibbs sampling, integrated likelihood and TVP Gibbs sampling guides, the UC and volatility tutorials |
| `ssm.mcse` | Computes the Monte Carlo standard error of a posterior mean | the Gibbs sampling, integrated likelihood and TVP Gibbs sampling guides |
| `ssm.geweke` | Computes the convergence diagnostic of Geweke (1992) | not yet in an example |
| `ssm.specvar0` | Estimates the long-run variance that the three diagnostics above use | through the diagnostics; the volatility tutorial |
| `ssm.models.tvp_midas` | Estimates the TVP-MIDAS regressions of Chan, Poon and Zhu (2026) and nowcasts | the MIDAS tutorial |
| `ssm.models.uc` | Estimates unobserved components models, among them UCSV | the UC tutorial |
| `ssm.models.uc_forecast` | Forecasts with the models of `ssm.models.uc` | the UC tutorial |
| `ssm.models.sv` | Estimates SV models whose errors can have an MA(1) or ARMA(1,1) term and Student-t tails | the volatility tutorial |
| `ssm.models.sv_forecast` | Forecasts the next value with the models of `ssm.models.sv` | the volatility tutorial |

The last five rows, in `ssm.models`, estimate complete models in one call or forecast with them.
Thirteen of the other functions have the same code as their counterparts in
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit);
[`tests/variant_map.md`](tests/variant_map.md) lists them. The Econometrics Toolbox has a class
named `ssm`: calls such as `ssm.simulate_states` resolve to this package, and `ssm(...)` still
constructs the toolbox's state space object.

## Verification

The unit tests check each function against the code it comes from or replaces. Where a function
replaces lines of a published script, the test runs the script as published and with the function
swapped in, and requires the same draws to rounding error. Functions with no published counterpart
are checked against a Kalman filter, dense algebra, simulation or a density computed on a grid.
[`tests/variant_map.md`](tests/variant_map.md) records where each function's code comes from,
which archived copies it matches and which pairs must never be merged.

On every push to `main` and every pull request, CI checks the archived packages against their
`as-published` tags and the copies in [`tests/fixtures/`](tests/fixtures/) against their md5s. It
also runs the unit suite, `setup.m`, the examples, each tutorial's `your_data.m` and each guide's
`guide.m`. To run the tests locally:

```matlab
run tests/unit/run_unit_tests.m
run tests/run_examples.m
```

## Related Repositories

[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) is the companion library for
Bayesian VARs. It also archives, as `chan_jeliazkov2009_statespace`, the replication package of
Chan and Jeliazkov (2009), which implements the precision sampler.

Every quarter, [trend-cycle-toolkit](https://github.com/joshuaccchan/trend-cycle-toolkit)
re-estimates published models of US trend inflation, the output gap and trend output growth, five
of them from packages archived here.

## Citation

Cite the paper behind each method you use: [`CITING.md`](CITING.md) maps every replication
package, library function and example to its paper, with BibTeX. To cite the toolkit itself:

> Chan, J. C. C. (2026). *statespace-toolkit: MATLAB code for Bayesian state space models*. Zenodo. https://doi.org/10.5281/zenodo.22884655

The DOI resolves to the latest release. `CITATION.cff` holds the machine-readable record that
GitHub's "Cite this repository" button reads.

## License

MIT; see [`LICENSE`](LICENSE). The license covers the archived packages too. The one third-party
file, `kde2d.m` in `grant_chan2017_jedc_hpfilter`, the kernel density estimator of Botev,
Grotowski and Kroese (2010), keeps its own terms. [`NOTICE.md`](NOTICE.md) states what the license
covers.
