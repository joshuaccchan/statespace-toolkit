# Variant map

For every function in `core/+ssm/`: where its code comes from, which legacy copies in this
repository it matches, and how that was verified. A legacy file with the same role that
computes something different is listed under **never merge** and keeps its own copy.

Verification: "code" = code-identical once comments and blank lines are removed, apart
from the differences named; "unit" = exact equality in `tests/unit/`, with stochastic
functions compared draw for draw under a fixed seed.

## Twins of bvar-toolkit Functions (18 September 2026)

Each twin is its bvar-toolkit original from the function line on, byte for byte, checked
by `tests/unit/test_twins.m` against the copies in `tests/fixtures/bvar-toolkit/` (bvar-toolkit
commit `b8f7021`, and `e412336` for `ksc_rw_h0` and `ksc_rw_diffuse`). Only the headers differ,
apart from `ssm.diffmat`'s error identifiers. Both libraries use the lower Cholesky factor throughout;
bvar-toolkit switched `ksc_rw_h0` to it in `d3b9494`, with bitwise the same draws. In
`fa17f41` it made `ksc_rw_h0` compute the mean with that factor, as `(Ch')\(Ch\b)`, which
changes the draws in the last bits, and `ssm.ksc_rw_h0` followed. `e412336` corrected the
comment on the draw of h in both.

| Function | Twin | Legacy copies in this repository | Verified |
|---|---|---|---|
| `ssm.surform` | `bvar.util.surform` | chan_clark_koop2018_jmcb_trendie `SURform.m` (code); chan2017_jbes_svm `SURform.m` and chan_grant2016_csda_dic `DIC/SURform.m` (`[r c]` for `[r,c]`); chan_eisenstat2015_er_mlce `ML_CE/SURform.m` (the same, and no closing `end`); fixture `chan-jeliazkov-2009/SURform.m` (a rewrite with an input check and an explicit size argument) | unit, all five (`test_surform`) |
| `ssm.tnormrnd` | `bvar.util.tnormrnd` | chan_clark_koop2018_jmcb_trendie, chan_grant2016_eneco_garchsv, chan_koop_potter2016_jae_boundedpc `tnormrnd.m` (code); grant_chan2017_jedc_hpfilter and grant_chan2017_jmcb_trendcycle `tnormrnd.m` (`\|\|` and `&&` for `\|` and `&` on scalar conditions); chan_eisenstat2015_er_mlce `ML_CE/tnormrnd.m` (the same, spacing, and no closing `end`) | unit, all six, draw for draw with scalar and vector arguments (`test_tnormrnd`) |
| `ssm.shaded_band` | `bvar.util.shaded_band` | none | unit (`test_shaded_band`) |
| `ssm.ksc_rw_h0` | `bvar.sv.ksc_rw_h0` | none; bvar-toolkit verifies its twin against five published `SVRW.m` copies | unit, draw for draw against the twin (`test_ksc_rw_h0`) |
| `ssm.ksc_rw_diffuse` | `bvar.sv.ksc_rw_diffuse` (fixture at bvar-toolkit `e412336`) | chan_clark_koop2018_jmcb_trendie `SVRW.m` with `h0 = 0`, the same draw with the mean solved by backslash. Never merge with `ssm.ksc_rw_h0`, which has another initial condition | unit (`test_ksc_rw_diffuse`), 22 September 2026: draw for draw against the twin; against `SVRW.m`, the same random number stream and the path to rounding |
| `ssm.diffmat` | `bvar.util.diffmat`, with error identifiers `ssm:diffmat:*` | new in bvar-toolkit; reproduces the inline spellings in chan2013_joe_masv `UC_MA.m` and `SV.m` and chan_grant2016_eneco_garchsv `loglike_garch_ma.m` | unit (`test_diffmat`) |

## Written Here

Functions not extracted from a package. Each is tested against the inline code it
generalizes, and none is substituted into the legacy-derived twins above.

| Function | Generalizes | Verified |
|---|---|---|
| `ssm.simulate_states` | the precision sampler as Algorithm 1 of Chan and Jeliazkov (2009), with the mean and the draw from one Cholesky factor; a more efficient form of the book's Algorithm 9.1. It replaces the posterior mean `K\c` and the draw `mu + chol(K,'lower')'\randn(n,1)` written inline in sp_code `UC.m` and `SVRW.m`, chan-jeliazkov-2009 `UC.m`, `linreg_tvp.m` and `DFM.m`, and the book's chapter 9 and 10 scripts, which factor K twice; for a block-banded K, MATLAB solves `K\c` by a banded LU | unit (`test_simulate_states`), 19 September 2026: the four anchor scripts, run whole on short chains with the mean line and the inline draw replaced by one call, leave the random number stream in the same state and store draws within 3.2e-13 of their scale (`linreg_tvp.m`; the others within 3.5e-14), the difference being the mean's solve; on a local level model, the mean returned equals `K\b` and the Kalman smoother's mean, `inv(K)` equals the smoother's covariances, and 20,000 draws match its means, variances and lag-one covariances |
| `ssm.intlike` | the integrated likelihood of Chan and Jeliazkov (2009), eq. (11), evaluated at the posterior mean; grant_chan2017_jmcb_trendcycle `intlike_UC0.m` and chan_grant2016_csda_dic `DIC/intlike_tvpvar.m` evaluate it in the equivalent form with `b'*inv(K)*b` | unit (`test_intlike`), 18 September 2026: the Kalman filter's prediction-error log likelihood on a local level model (difference 8e-13), a TVP regression (9e-13) and the UC model with an AR(2) cycle on the trendcycle package's GDP data (4e-12); `intlike_tvpvar` on the DIC package's data (4e-12); `intlike_UC0` on the GDP data (2.6e-8, rounding in the legacy form, whose terms near 1e7 cancel on data near 750; the Kalman filter agrees with `ssm.intlike`) |
| `ssm.mode_newton` | the Newton-Raphson loops of three accept-reject MH steps: the book's `sample_SVM_h_ARMH.m` (step `h + K\grad`, tolerance 1e-4, cap 100) and the published `sample_CSV.m` (chan2023_joe_mlvarsv) and `UC_SVM.m` (chan2017_jbes_svm), whose step is `K\(K*h + grad)` spelled out, with tolerance 1e-3 and no cap | unit (`test_mode_newton`): one step to the mean of a Gaussian, zero gradient at the mode of the CSV target, and the cap and a NaN step raise. Its step is the book's, so against the published loops the mode agrees to rounding (`test_armh`). |
| `ssm.lagpolymat` | the lag polynomial matrices I - phi_1*L - ... - phi_p*L^p built inline: the second-difference matrix and the AR(2) matrix of the book's `chapter09/UC_output_gap.m`, the AR(2) matrix of chan2018_er_spectest `TV_NAIRU_AR2.m`, and `ssm.diffmat` for p = 1. The SV(2) scripts of chan_grant2016_eneco_garchsv and chan_grant2016_jfec_dicsv build a different matrix, whose second row has no phi term because the first two states are initial values; they are not generalized here | unit (`test_lagpolymat`), 19 September 2026: bitwise equal to each spelling above and to `ssm.diffmat(T, a)` for a random walk, an AR(1) and an MA(1) transform; `H*x` equals `filter([1 -phi], 1, x)`; lags beyond the path drop out |
| `ssm.armh` | the accept-reject Metropolis-Hastings steps of the same three implementations: Newton mode, Gaussian proposal, screening with `c = c_reject*f(mode)/g(mode)`, `c_reject = 3` (`kappa` in the book's code), and the MH correction, with random numbers drawn in the same order | unit (`test_armh`), 18 September 2026: 40 sweeps of the book's function give bitwise the same draws, acceptances and random number stream; 50 sweeps of `sample_CSV.m` and 150 of `UC_SVM.m` run whole give the same acceptances and stream, with draws within 8.9e-16 and 3.5e-13; Geweke tests at `c_reject` = 0.2, 1, 3 and 20 and with forced accepts. |
| `ssm.select_obs` | the split of stacked data into observed and missing values, `y = So*yo + Sm*ym`, written from Section 2.1 of Chan, Poon and Zhu (2023), whose code is not archived here; twin: `bvar.util.select_obs` in bvar-toolkit, copied from here | unit (`test_select_obs`), 19 September 2026: both illustrations of that section reproduced exactly; `[So, Sm]` a permutation matrix; the conditional mean and precision of the missing values of a VAR(1) with a hole and a ragged edge equal to dense Gaussian conditioning |

## Never Merge

| File | Why it is not `ssm.ksc_rw_h0` |
|---|---|
| chan_clark_koop2018_jmcb_trendie `legacy/SVRW.m` | `SVRW(Ystar,h,sig,h0,Vh)` takes a fifth argument, and its code differs in 19 lines |
| fixture `bvar-toolkit/.../sp_code/SVRW.m` | `[h S] = SVRW(ystar,h,omega2h,Vh)` has no `h0`, also returns the mixture indicators, and its code differs in 26 lines |
