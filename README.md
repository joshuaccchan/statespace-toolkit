# statespace-toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22884655.svg)](https://doi.org/10.5281/zenodo.22884655)

MATLAB code for Bayesian state space models by [Joshua Chan](https://joshuachan.org). The library
under `core/` holds the building blocks of their samplers, from the precision sampler and the
integrated likelihood to the draw of missing data and an accept-reject Metropolis-Hastings step
for states whose conditional distribution is not Gaussian. Eight examples use them in complete
samplers. The repository also archives fourteen replication packages from
[joshuachan.org](https://joshuachan.org/code.html), each archived file byte for byte as
published.

```matlab
run setup.m                 % adds core/ to the path
cd examples
ex01_precision_sampler      % the precision sampler on a local level model
```

The code needs MATLAB with the Statistics and Machine Learning Toolbox. `setup.m` also checks for
the Optimization Toolbox, which four of the archived packages use.

## Learn the Methods

Most of the methods are developed in the book *Bayesian Macroeconometrics: Methods and
Applications* (Chapman & Hall/CRC, forthcoming): see the
[sample chapters](https://joshuachan.org/papers/BayesMacroBook_sample.pdf) and
[its code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics), with MATLAB,
R and Python for all fourteen chapters. The missing-data draw of ex05 comes from Chan, Poon and
Zhu (2023), and the TVP-MIDAS model of ex08 from Chan, Poon and Zhu (2026).

The eight scripts in [`examples/`](examples/) each run in under a minute. Start with ex01; ex02
compares the local level model of ex01 with one that adds an AR(1) transitory component. ex04
rewrites the book's `chapter09/UC_output_gap.m` with the library functions, and ex07 estimates a
simpler version of the model of Chan (2017) with the sampler of its `UC_SVM.m`. ex08 imposes its
linear restriction by the same update of an unconstrained draw that ex05 uses for the quarterly
aggregation.

| Script | What it shows | Data |
|---|---|---|
| `ex01_precision_sampler` | The precision sampler, `ssm.simulate_states`, on a local level model, checked against a Kalman smoother and the truth | generated; US CPI inflation |
| `ex02_integrated_likelihood` | Two models compared by marginal likelihood, with the states integrated out by `ssm.intlike`, checked against a Kalman filter and quadrature | US PCE inflation |
| `ex03_tvp_regression` | A time-varying parameter Phillips curve, and the precision sampler timed against the Kalman filter with backward sampling | US PCE inflation and output gap; generated |
| `ex04_output_gap` | The output gap from a local linear trend with an AR(2) cycle, built with `ssm.lagpolymat` | US real GDP |
| `ex05_missing_data` | Missing data and mixed frequencies with `ssm.select_obs`: a ragged edge, checked against dense algebra and the values removed, and monthly GDP estimated from quarterly GDP and monthly indicators | generated; FRED-MD and US real GDP |
| `ex06_dynamic_factor` | A dynamic factor model with one factor, as a business-cycle indicator | FRED-MD |
| `ex07_svm_armh` | Stochastic volatility in mean, with the log-volatility drawn by accept-reject Metropolis-Hastings, `ssm.armh` | US CPI inflation |
| `ex08_tvp_midas` | A MIDAS regression with time-varying weights under a linear restriction, checked against dense algebra and the truth | generated |

[`examples/data/README.md`](examples/data/README.md) gives the source of every data file and the
rows each example reads.

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
directly, or copy the example closest to your model; every example builds a complete sampler from
them.

| Function | Does | Shown in |
|---|---|---|
| `ssm.simulate_states` | Draws a state path from its posterior, given the banded precision matrix of the states: the precision sampler of Chan and Jeliazkov (2009) | ex01-ex08 |
| `ssm.intlike` | Computes the log likelihood of a linear Gaussian state space model, with the states integrated out | ex02 |
| `ssm.select_obs` | Splits the stacked data into observed and missing values, keeping the precision matrix of the missing values banded | ex05 |
| `ssm.diffmat` | Builds the first-difference matrix of a state equation, I - aL | ex01-ex03, ex06-ex08 |
| `ssm.lagpolymat` | Builds the matrix of a lag polynomial, such as second differences or an AR(2) | ex04 |
| `ssm.surform` | Builds the design matrix of a regression whose coefficients vary over time | ex03, ex07, ex08 |
| `ssm.mode_newton` | Finds the mode of a concave log density by Newton-Raphson, with a banded Hessian | through `ssm.armh` |
| `ssm.armh` | Takes the accept-reject Metropolis-Hastings step of Chan (2017), for states whose conditional distribution is not Gaussian | ex07 |
| `ssm.ksc_rw_h0` | Draws the log-volatility path of a random walk by the auxiliary mixture sampler of Kim, Shephard and Chib (1998) | ex08 |
| `ssm.tnormrnd` | Draws from a truncated normal distribution | ex02 |
| `ssm.shaded_band` | Shades credible bands in figures | ex01, ex03, ex04, ex07 |

Five of these functions, `ssm.diffmat`, `ssm.ksc_rw_h0`, `ssm.shaded_band`, `ssm.surform` and
`ssm.tnormrnd`, have the same code as their counterparts in
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit), apart from the error identifiers of
`ssm.diffmat`. A unit test checks each against a pinned copy of the bvar-toolkit original. The
Econometrics Toolbox has a class named `ssm`: calls such as `ssm.simulate_states` resolve to this
package, and `ssm(...)` still constructs the toolbox's state space object.

## Verification

The unit tests check each function against the code it comes from or replaces. `ssm.surform` and
`ssm.tnormrnd` are run beside every copy in the archived packages, four of `SURform.m` and six of
`tnormrnd.m`, and must give identical output,
`ssm.tnormrnd` draw for draw under a fixed seed. For `ssm.simulate_states`, the tests run the
published scripts `UC.m`, `linreg_tvp.m` and `DFM.m` on a short chain twice: once as published,
and once with the lines the function replaces swapped for a call to it. `UC_SVM.m` is run the same
way for `ssm.armh`, and the published functions `sample_CSV.m` and the book's
`sample_SVM_h_ARMH.m` are run on generated data beside `ssm.armh`, with the same target. Each of
these tests checks that the random number stream ends in the same state and that the draws agree
to rounding error.

`ssm.intlike` is checked against a Kalman filter and against two archived functions that compute
it, `intlike_UC0.m` and `intlike_tvpvar.m`. `ssm.diffmat` and `ssm.lagpolymat` are checked against the matrices the published
scripts write out, and `ssm.select_obs` against the worked illustrations of Chan, Poon and Zhu
(2023).

Files from other repositories that the tests run are held verbatim in
[`tests/fixtures/`](tests/fixtures/), with their md5s. On every push to `main` and every pull
request, CI checks the archived packages against their `as-published` tags and the fixtures
against their md5s, and runs the unit suite, `setup.m` and the examples. To run the tests
locally:

```matlab
run tests/unit/run_unit_tests.m
run tests/run_examples.m
```

[`tests/variant_map.md`](tests/variant_map.md) records where each function's code comes from,
which archived copies it matches and which pairs must never be merged.

## Related Repositories

[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) is the companion library for
Bayesian VARs. It also archives, as `chan_jeliazkov2009_statespace`, the replication package of
Chan and Jeliazkov (2009), which implements the precision sampler. The tests here run copies of
its `UC.m` and `SVRW.m`.

Every quarter, [trend-cycle-toolkit](https://github.com/joshuaccchan/trend-cycle-toolkit)
re-estimates published models of US trend inflation, the output gap and trend output growth. Five
of those models come from packages archived here: Chan (2013); Chan, Koop and Potter (2013);
Grant and Chan (2017, JMCB); Grant and Chan (2017, JEDC); and Chan, Clark and Koop (2018).

## Citation

Cite the paper behind each method you use. The table in
[Which Model Do I Want?](#which-model-do-i-want) and [`provenance.md`](provenance.md) give the
paper for each package. The header of each function and example lists the works it draws on. For
the library:

- `ssm.simulate_states` and `ssm.intlike`: Chan, J.C.C. and Jeliazkov, I. (2009). Efficient
  Simulation and Integrated Likelihood Estimation in State Space Models, *International Journal of
  Mathematical Modelling and Numerical Optimisation*, 1(1/2): 101-120.
- `ssm.armh`: Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
  Parameters: An Application to Inflation Modeling, *Journal of Business and Economic
  Statistics*, 35(1): 17-28.
- `ssm.select_obs`: Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally
  Gaussian State Space Models with Missing Data, *Journal of Econometrics*, 236(1): 105468.
- `ssm.ksc_rw_h0`: Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
  Inference and Comparison with ARCH Models, *Review of Economic Studies*, 65(3): 361-393.
- The other functions: Chan, J.C.C. (forthcoming). *Bayesian Macroeconometrics: Methods and
  Applications*, Chapman & Hall/CRC.

To cite the toolkit itself:

> Chan, J. C. C. (2026). *statespace-toolkit: MATLAB code for Bayesian state space models*. Zenodo. https://doi.org/10.5281/zenodo.22884655

The DOI resolves to the latest release. `CITATION.cff` holds the machine-readable record that
GitHub's "Cite this repository" button reads.

## License

MIT; see [`LICENSE`](LICENSE). The license covers the archived packages too. Third-party files
keep their own licenses, among them `kde2d.m` in `grant_chan2017_jedc_hpfilter`, the kernel
density estimator of Botev, Grotowski and Kroese (2010). [`NOTICE.md`](NOTICE.md) states what the
license covers and that citing the papers is scholarly practice, which the license does not
require.
