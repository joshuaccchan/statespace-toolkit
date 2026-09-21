# Golden-Run Manifest

One row per entry script, run through `run_golden.ps1` on R2025b Update 4, 19 September
2026. **Anchor** says what a later driver must match: *bitwise* where the run is
reproducible, because the script never seeds (a fresh `matlab -batch` session starts from
MATLAB's default stream) or seeds with a constant; *summary* where the script seeds from
the clock and only Monte Carlo agreement is possible. Two fresh-session runs of `UC_SVM.m`
gave identical draws in every stored array. Scripts with a model selector are captured at
the selection they ship with. A `savegolden` capture is the same script with a numeric
capture appended (see `README.md`); its values are listed where the as-shipped log holds
none.

## Estimation Scripts

| Package | Entry | Anchor | Runtime | Captured values |
|---|---|---|---|---|
| `chan2017_jbes_svm` | `UC_SVM.m` | bitwise | 48 s | its table: mu 0.524, beta 0.001, phi 0.962, sigma2 0.082, omega2_alpha 0.039, omega_{alpha,tau} 0.009, omega2_tau 0.117 |
| `chan_koop_potter2013_jbes_trendbound` | `ARtrend_bound.m` | summary | 63 s | progress only. savegolden: (sigtau, sigrho, sigh) means 0.0215, 0.00116, 0.0637; tau, rho and h paths in `golden_capture.mat` |
| `chan2013_joe_masv` | `main_UCMA.m` | summary | 25 s | model 4, AR(2)-MA, as shipped; progress only. savegolden: stheta means 1.087, 0.965, 0.0806; sbeta 0.290, 0.928, -0.018; psi -0.390; h path |
| `chan_hsiao2014_wiley_sv` | `plainSV.m` | summary | 22 s | progress only. savegolden: (mu, muh, phih, sigh2) means -0.0292, -0.750, 0.989, 0.0175 |
| `chan_hsiao2014_wiley_sv` | `MASV.m` | summary | 104 s | progress only. savegolden: (mu, muh, phih, sigh2) means -0.0085, -2.189, 0.983, 0.0173; psi 0.141 |
| `chan_hsiao2014_wiley_sv` | `MASVt.m` | summary | 19 s | progress only. savegolden: store_theta means -0.088, 6.348, 0.126, 1.093, 0.985, 0.0159 |
| `chan_grant2016_csda_dic` | `SF.m` | bitwise | 106 s | progress only; the clock only times the run. savegolden: DIC_choice 5, the complete-data DIC5, 26344.59 (NSE 22.57) |
| `chan_grant2016_csda_dic` | `VAR1.m` | bitwise | 60 s | progress only. savegolden: observed-data DIC 3009.84 (NSE 0.50) |
| `chan_grant2016_csda_dic` | `TVPVAR.m` | bitwise | 28 min | progress only. savegolden: DIC_choice 2, the observed-data DIC2, 3203.46 (NSE 1.27) |
| `chan_grant2016_csda_dic` | `CTVPVAR.m` | bitwise | 17 min | progress only. savegolden: DIC_choice 2, the observed-data DIC2, 3002.38 (NSE 1.64) |
| `chan_grant2016_csda_dic` | `semireg.m` | none | n/a | not runnable: its BMI data are restricted, as the package README says |
| `chan_eisenstat2015_er_mlce` | `probit_mc.m`, `logit_mc.m`, `bin_t_mc.m` | summary | 1-4 s | log ML (NSE): -140.669 (0.0010), -141.236 (0.0011), -141.075 (0.0021); `logit_mc.m` also prints fminunc's "local minimum possible" message |
| `chan_eisenstat2015_er_mlce` | `VAR_mc.m`, `TVPVAR_mc.m`, `DFVAR_mc.m` | summary | 5 s, 17 min, 45 s | log ML (NSE): -905.370 (0.0016), -899.823 (0.0551), -903.653 (0.0853) |
| `chan_grant2016_jfec_dicsv` | `main_SV.m` | summary | 11 min | its table: mu 0.00, mu_h -9.11, phi_h 0.98, omega2_h 0.04; observed-data DIC -9082.3 (0.35), effective number of parameters 9.7 (0.34) |
| `chan_grant2016_eneco_garchsv` | `main_SV.m` | summary | 53 s | with the `autocorr` patch. Its table: mu 0.11, mu_h 2.63, phi_h 0.97, omega2_h 0.03; log ML -2612.3 (0.02) |
| `chan_grant2016_eneco_garchsv` | `main_GARCH.m` | summary | 26 s | with the `autocorr` patch. Its table: mu 0.14, alpha_0 0.21, alpha_1 0.09, beta_1 0.91; log ML -2652.0 (0.01) |
| `chan2018_er_spectest` | `main_NAIRU.m` | summary | 2.6 min | its table: lambda -0.59, phi_1 1.64, phi_2 -0.70, sigma2_u 0.07, omega2_h 0.27, omega2_g 0.11, omega2_nu 0.01; log BF 2.1 (0.43). Then errors at `legend('posterior','prior',1)`, a positional location R2025b rejects, after printing everything |
| `chan2018_er_spectest` | `main_UCSV.m` | summary | 10 min | reads `OECD_G7CPI.xls` with `xlsread` and a range, which needs Excel. Its table: omega2_h 0.11, omega2_g 0.11; log BF_uh 81.9 (6.10), BF_ug 13.9 (3.04), BF_u,gh 204.9 (4.97). Then errors at the same `legend` call |
| `chan_clark_koop2018_jmcb_trendie` | `main_estimation.m` | summary | 4.1 min | model 1 (M1) as shipped; `xlsread` with a range (Excel); progress only. savegolden: store_theta means (11); pistar, d, b, lamv and lamn paths |
| `chan_koop_potter2016_jae_boundedpc` | `biUC.m` | summary | 9.1 min | prints the model specification only. savegolden: rhou means 1.619, -0.676; sig2 means 0.110, 0.107, 0.024, 0.0075, 0.0023, 0.0018; the trend, NAIRU, lambda and volatility paths |
| `chan_song2018_jmcb_inflrv` | `main_inflation_RV.m` | bitwise | 31 s | model 1 (UCSV-RV) as shipped, which never seeds (the scripts of models 2-8 call `rng(123)`); progress only. savegolden: sigma2h 0.275, sigma2g 0.104, sigma2z 0.661, a (-0.453, 1.092), g1 -1.901, h1 2.298 |
| `grant_chan2017_jedc_hpfilter` | `main_script.m` | summary | 76 s | its table: phi_1 1.31, phi_2 -0.37, sigma^2_c 0.76, sigma^2_tau 0.00, rho -0.01; log ML -369.8 (0.03) |
| `grant_chan2017_jmcb_trendcycle` | `main_UC.m` | summary | 66 s | its table: mu 0.78, phi_1 0.97, phi_2 -0.38, sigma^2_c 1.08, sigma^2_tau 1.79, rho -0.87; log ML -365.0 (0.04) |

## Forecasting Scripts

Each runs its recursive exercise for the model selected as shipped.

| Package | Entry | Anchor | Runtime | Captured values |
|---|---|---|---|---|
| `chan_grant2016_eneco_garchsv` | `main_forecasting.m` | summary | 48 min | model 1 (SV), 782 forecast origins, with the `autocorr` patch; log predictive score -2160.5 |
| `chan_clark_koop2018_jmcb_trendie` | `main_forecasting.m` | summary | 3.4 h | model 1 (M1) as shipped, 125 forecast origins; `xlsread` with a range (Excel). Its table of RMSFE and log predictive likelihood: 1Q 2.04, -170.56; 2Q 1.64, -156.25; 4Q 1.16, -137.45; 8Q 1.14, -140.84; 12Q 1.16, -143.36; 16Q 1.18, -143.84; 20Q 1.21, -146.08; 6-10Y 0.91, -112.31 |
| `chan_koop_potter2016_jae_boundedpc` | `main_forecasting.m` | summary | 18.0 h | model 1 (Bi-UC) as shipped, 168 forecast origins. RMSFE for CPI inflation and the unemployment rate: 1Q 2.07, 0.25; 4Q 2.63, 1.01; 8Q 3.07, 1.57; 12Q 3.08, 1.74; 16Q 3.19, 1.79. The log predictive likelihood table is headed `Joint, CPI, Urate`, but `forecast_biUC.m` stores the densities as CPI, unemployment rate and joint, and the columns print in that order: 1Q -307, -14, -321; 4Q -342, -219, -565; 8Q -359, -293, -667; 12Q -362, -312, -697; 16Q -371, -317, -715 |
