# Tutorials

Each tutorial answers one empirical question with the models of published papers, on the data in
[`examples/data/`](../examples/data/). A `build.m` regenerates every figure and number
on its page, and a `your_data.m` runs the same analysis on another series.

The paper column links the published version; each page gives the full references.

| Tutorial | Code | Paper |
|---|---|---|
| [Which Unobserved Components Model Should I Use for Inflation?](uc_specification/) | `ssm.simulate_states`, `ssm.ksc_rw_h0`, `ssm.ksc_rw_diffuse`, `ssm.ksc_ar1_mean`, `ssm.diffmat`, `ssm.surform`, `ssm.tnormrnd`; the archive `chan_koop_potter2013_jbes_trendbound` | [Chan (2013)](https://doi.org/10.1016/j.jeconom.2013.05.003), [Chan, Koop and Potter (2013)](https://doi.org/10.1080/07350015.2012.741549), [Chan, Clark and Koop (2018)](https://doi.org/10.1111/jmcb.12452), [Stock and Watson (2007)](https://doi.org/10.1111/j.1538-4616.2007.00014.x) |
