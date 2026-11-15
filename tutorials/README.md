# Tutorials

Each tutorial answers one empirical question with the models of published papers, on the data in
[`examples/data/`](../examples/data/). A `build.m` regenerates every figure and number
on its page, and a `your_data.m` runs the same analysis on another series.

The paper column links the published version; each page gives the full references.

| Tutorial | Code | Paper |
|---|---|---|
| [Which Unobserved Components Model Should I Use for Inflation?](uc_specification/) | `ssm.models.uc`, `ssm.models.uc_forecast`, `ssm.simulate_states`, `ssm.ksc_rw_h0`, `ssm.ksc_rw_diffuse`, `ssm.ksc_ar1_mean`, `ssm.diffmat`, `ssm.surform`, `ssm.tnormrnd`; the archive `chan_koop_potter2013_jbes_trendbound` | [Chan (2013)](https://doi.org/10.1016/j.jeconom.2013.05.003), [Chan, Koop and Potter (2013)](https://doi.org/10.1080/07350015.2012.741549), [Chan, Clark and Koop (2018)](https://doi.org/10.1111/jmcb.12452), [Stock and Watson (2007)](https://doi.org/10.1111/j.1538-4616.2007.00014.x) |
| [Which Parameters of a GDP Nowcasting Model Should Vary over Time?](midas_nowcast/) | `ssm.models.tvp_midas`, `ssm.midas_basis`, `ssm.simulate_states`, `ssm.restrict`, `ssm.ksc_rw_diffuse`, `ssm.diffmat`, `ssm.surform`, `ssm.shaded_band` | [Chan, Poon and Zhu (2026)](https://doi.org/10.1016/j.jeconom.2025.106090) |
| [Which Stochastic Volatility Model Forecasts Best?](sv_volatility/) | `ssm.models.sv`, `ssm.models.sv_forecast`, `ssm.ksc_ar1_mean`, `ssm.ksc_rw_diffuse`, `ssm.sv_params`, `ssm.nu_studentt`, `ssm.ma_params` | [Chan and Grant (2016)](https://doi.org/10.1016/j.eneco.2015.12.003), [Chan (2013)](https://doi.org/10.1016/j.jeconom.2013.05.003), [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034) |

The tutorials of this repository and of
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials) are listed
together, grouped by topic, at [joshuachan.org/tutorials](https://joshuachan.org/tutorials/).
