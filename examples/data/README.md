# Example Data

| File | Contents | Source |
|---|---|---|
| `USCPI.csv` | US CPI inflation, monthly, 1948M1-2024M12: 1200 times the log change in the consumer price index (column `CPIAUCSL`). ex01 reads rows 2-865, 1948M1-2019M12, as `UC.m` does | FRED series CPIAUCSL; the file of the chan-jeliazkov-2009 repository at commit `84c9fac`, where `UC.m` reads it |
| `USGDP.csv` | US real GDP, quarterly, 1947Q1-2024Q4, billions of chained 2017 dollars (column `GDPC1`). ex04 reads rows 2-293, 1947Q1-2019Q4, and takes 100 times the log, as `UC_output_gap.m` does | FRED series GDPC1; the file of the bayesian-macroeconometrics repository at commit `1e16335`, where `chapter09/UC_output_gap.m` reads it |
| `USPCE_OutputGap.csv` | US PCE inflation, quarterly, 1960Q1-2024Q4: 400 times the log change in the PCE price index (column `PCECTPI`), with the output gap for 1960Q1-2019Q4 (column `Output Gap`). ex02 reads the inflation column, rows 2-261 | FRED series PCECTPI; the file of the chan-jeliazkov-2009 repository at commit `84c9fac`, where `linreg_tvp.m` reads it |
