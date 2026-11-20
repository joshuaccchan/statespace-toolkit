# Examples

Each script runs in under a minute. The table lists what each one shows and the data it uses.

| Script | What it shows | Data |
|---|---|---|
| `ex01_precision_sampler` | The precision sampler, `ssm.simulate_states`, on a local level model, checked against a Kalman smoother and the truth | generated; US CPI inflation |
| `ex02_integrated_likelihood` | Two models compared by marginal likelihood, with the states integrated out by `ssm.intlike`, checked against a Kalman filter and quadrature | US PCE inflation |
| `ex03_tvp_regression` | A time-varying parameter Phillips curve, and the precision sampler timed against the Kalman filter with backward sampling | US PCE inflation and output gap; generated |
| `ex04_output_gap` | The output gap from a local linear trend with an AR(2) cycle, built with `ssm.lagpolymat` | US real GDP |
| `ex05_missing_data` | Missing data and mixed frequencies with `ssm.select_obs`: the selection matrices of a small pattern, then a late start, a hole and a ragged edge in one VAR, and a series observed only through quarterly aggregates, both checked against dense algebra and the values removed | generated |
| `ex06_dynamic_factor` | A dynamic factor model with one factor, as a business-cycle indicator | FRED-MD |
| `ex07_aux_mixture` | The auxiliary mixture sampler for stochastic volatility, `ssm.ksc_rw_h0`: the seven-component mixture against the exact density it approximates, the collapsed Gibbs sampler checked against the truth, and the draws reweighted to the exact posterior | generated |
| `ex08_sv_ma_t` | Stochastic volatility with MA(1) and Student-t errors, the models of Chan and Hsiao (2014) with mean zero: the log-volatility drawn by `ssm.ksc_ar1_mean`, and the MA coefficient and the degrees of freedom by Metropolis-Hastings with normal proposals at modes found by `ssm.mode_newton` | silver returns |
| `ex09_svm_armh` | Stochastic volatility in mean, with the log-volatility drawn by accept-reject Metropolis-Hastings, `ssm.armh` | US CPI inflation |
| `ex10_tvp_midas` | A MIDAS regression with time-varying weights under a linear restriction, checked against dense algebra and the truth | generated |

[`data/README.md`](data/README.md) gives the source of every data file and the rows each
example reads.
