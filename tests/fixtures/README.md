# Fixtures

Verbatim copies of files from other repositories, which the equivalence tests run
unmodified. Each file sits at its path in the source repository, under a folder named
for that repository, and is byte-identical to the file committed at the commit below.
`MD5SUMS` holds their md5s; CI checks every file here against it and fails on a file
with no entry.

| Source repository | Commit | Files | What they are |
|---|---|---|---|
| [bayesian-macroeconometrics](https://github.com/joshuaccchan/bayesian-macroeconometrics) | `21850cdc0ba041e7798f425ae21423d77a7d64c1` | `code/matlab/chapter10/sample_SVM_h_ARMH.m` | the book's accept-reject Metropolis-Hastings step for the log-volatility of the SVM model |
| [bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) | `cadbee8f015dc3784e338f91481cc7e5a7ccb301` (tag `as-published/chan2023_joe_mlvarsv`) | `replications/chan2023_joe_mlvarsv/legacy/utility/sample_CSV.m` | the published accept-reject Metropolis-Hastings step for common stochastic volatility, from Chan (2023) |
| [bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) | `d8763165c60e6ec1ecd0ee3fe0f6bf8adad32519` (tag `as-published/chan_jeliazkov2009_statespace`) | `replications/chan_jeliazkov2009_statespace/legacy/sp_code/UC.m`, `SVRW.m`, `USCPI.csv` | the published precision sampler of Chan and Jeliazkov (2009), with the log-volatility sampler and the data it uses |
| [bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) | `b8f702100e4be8e81175297d462f0eb38a3e274d` | `core/+bvar/+util/surform.m`, `tnormrnd.m`, `shaded_band.m`, `diffmat.m` | four of the five bvar-toolkit functions that have code-identical twins in `core/+ssm/` |
| [bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) | `fa17f4172f80b8188ac230c8e22923e66594681e` | `core/+bvar/+sv/ksc_rw_h0.m` | the fifth, the twin of `ssm.ksc_rw_h0`, pinned to the commit that switched its mean to the Cholesky factor it draws with |
| [chan-jeliazkov-2009](https://github.com/joshuaccchan/chan-jeliazkov-2009) | `84c9facb5baa193e75c9aa068274035ffdc095dd` | `UC.m`, `linreg_tvp.m`, `DFM.m`, `SURform.m`, `USCPI.csv`, `USPCE_OutputGap.csv`, `FRED-MD.csv` | three worked examples of the precision sampler, with the helper and the data they read |

The copies were taken with `git show <commit>:<path>`, so they carry the committed bytes.
chan-jeliazkov-2009 has no `.gitattributes`, and a Windows checkout of it has CRLF line
endings in its working tree where the committed files, and these copies, have LF.
