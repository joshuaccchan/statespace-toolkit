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
commit `947df3b`). Only the headers differ, apart from `ssm.diffmat`'s error identifiers
and `ssm.ksc_rw_h0`'s Cholesky factor: this library uses the lower factor throughout.

| Function | Twin | Legacy copies in this repository | Verified |
|---|---|---|---|
| `ssm.surform` | `bvar.util.surform` | chan_clark_koop2018_jmcb_trendie `SURform.m` (code); chan2017_jbes_svm `SURform.m` and chan_grant2016_csda_dic `DIC/SURform.m` (`[r c]` for `[r,c]`); chan_eisenstat2015_er_mlce `ML_CE/SURform.m` (the same, and no closing `end`); fixture `chan-jeliazkov-2009/SURform.m` (a rewrite with an input check and an explicit size argument) | unit, all five (`test_surform`) |
| `ssm.tnormrnd` | `bvar.util.tnormrnd` | chan_clark_koop2018_jmcb_trendie, chan_grant2016_eneco_garchsv, chan_koop_potter2016_jae_boundedpc `tnormrnd.m` (code); grant_chan2017_jedc_hpfilter and grant_chan2017_jmcb_trendcycle `tnormrnd.m` (`\|\|` and `&&` for `\|` and `&` on scalar conditions); chan_eisenstat2015_er_mlce `ML_CE/tnormrnd.m` (the same, spacing, and no closing `end`) | unit, all six, draw for draw with scalar and vector arguments (`test_tnormrnd`) |
| `ssm.shaded_band` | `bvar.util.shaded_band` | none | unit (`test_shaded_band`) |
| `ssm.ksc_rw_h0` | `bvar.sv.ksc_rw_h0`, with `chol(Ph,'lower')` and `Ch'\randn(T,1)` where the twin has `chol(Ph)` and `Ch\randn(T,1)` | none; bvar-toolkit verifies its twin against five published `SVRW.m` copies | unit, draw for draw against the twin, so the lower factor gives bitwise the same draws (`test_ksc_rw_h0`) |
| `ssm.diffmat` | `bvar.util.diffmat`, with error identifiers `ssm:diffmat:*` | new in bvar-toolkit; reproduces the inline spellings in chan2013_joe_masv `UC_MA.m` and `SV.m` and chan_grant2016_eneco_garchsv `loglike_garch_ma.m` | unit (`test_diffmat`) |

## Written Here

Functions not extracted from a package. Each is tested against the inline code it
generalizes, and none is substituted into the legacy-derived twins above.

| Function | Generalizes | Verified |
|---|---|---|
| `ssm.simulate_states` | the precision sampler, Algorithm 9.1 of the book: the draw `mu + chol(K,'lower')'\randn(n,1)` written inline, after the posterior mean, in sp_code `UC.m` and `SVRW.m`, chan-jeliazkov-2009 `UC.m`, `linreg_tvp.m` and `DFM.m`, and the book's chapter 9 and 10 scripts | unit (`test_simulate_states`): the four anchor scripts, run whole on short chains with the inline draw replaced by the call and their own posterior mean kept, store identical draws; on a local level model, `K\b` and `inv(K)` equal the Kalman smoother's moments, and 20,000 draws match its means, variances and lag-one covariances. The upper-factor spelling `chol(K)\randn(n,1)`, which `bvar.sv.ksc_rw_h0` uses, gives bitwise the same draws on all four anchors. |
| `ssm.intlike` | the integrated likelihood of Chan and Jeliazkov (2009), eq. (11), evaluated at the posterior mean; grant_chan2017_jmcb_trendcycle `intlike_UC0.m` and chan_grant2016_csda_dic `DIC/intlike_tvpvar.m` evaluate it in the equivalent form with `b'*inv(K)*b` | unit (`test_intlike`), 18 September 2026: the Kalman filter's prediction-error log likelihood on a local level model (difference 8e-13), a TVP regression (9e-13) and the UC model with an AR(2) cycle on the trendcycle package's GDP data (4e-12); `intlike_tvpvar` on the DIC package's data (4e-12); `intlike_UC0` on the GDP data (2.6e-8, rounding in the legacy form, whose terms near 1e7 cancel on data near 750; the Kalman filter agrees with `ssm.intlike`) |

## Never Merge

| File | Why it is not `ssm.ksc_rw_h0` |
|---|---|
| chan_clark_koop2018_jmcb_trendie `legacy/SVRW.m` | `SVRW(Ystar,h,sig,h0,Vh)` takes a fifth argument, and its code differs in 19 lines |
| fixture `bvar-toolkit/.../sp_code/SVRW.m` | `[h S] = SVRW(ystar,h,omega2h,Vh)` has no `h0`, also returns the mixture indicators, and its code differs in 26 lines |
