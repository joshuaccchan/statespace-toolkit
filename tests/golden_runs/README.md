# Golden Runs

Each archived package is run once on current MATLAB (R2025b), exactly as published, and
what it prints and saves is captured as a *golden*: the reference that any later
functionized driver or library replacement must reproduce, bitwise where the run is
reproducible and within Monte Carlo error where it is not.

## Protocol

1. `run_golden.ps1 -Slug <slug> -Entry <script.m>` does, in order:
   - copies `replications/<slug>/legacy/` to
     `%LOCALAPPDATA%\statespace-toolkit\golden_runs\<slug>_<time>\`, outside the
     repository, because a sync client locks freshly written files and would sync the
     MCMC scratch output;
   - overlays `tests/golden_runs/patches/<slug>/` if that folder exists;
   - finds the entry script in the copy, runs it from its own folder under
     `matlab -batch`, and logs what it prints;
   - copies the log and every file the run created or changed into
     `tests/golden/<slug>/<entry>_<date>/`.
2. **Legacy folders are never edited.** Patches apply to the build copy only, each is as
   small as running the script requires, and each patched or added file opens with a
   comment saying what it changes and why. A `-PatchDir` and `-Label` pair runs a
   variant, such as a script whose results go only to figures with a numeric capture
   appended (`<slug>-savegolden`).
3. Most packages seed the generator from the clock (`rand('state', sum(100*clock))` and
   the like), so their goldens anchor summary statistics, within Monte Carlo error. A
   script that never seeds starts from MATLAB's default stream in a fresh `matlab -batch`
   session, and one that calls `rng` with a constant is reproducible as well; those
   goldens are bitwise anchors. `manifest.md` says which is which for every capture.
4. `manifest.md` lists every entry script with its toolboxes, runtime, blockers and
   capture status.

## Patches

| Folder | Adds or changes | Why |
|---|---|---|
| `chan_grant2016_eneco_garchsv/autocorr.m` | adds a function, written here | the scripts call `autocorr(u, nlag)`, the positional form R2025b's Econometrics Toolbox rejects; the function returns what that call returned, and equals the 2003 MathWorks `autocorr` bit for bit on the package's data and on generated series |
| `<slug>-savegolden/...` (12 scripts in 7 packages) | appends a numeric capture to the entry script | the script's results go only to figures or stay in the workspace, so the as-shipped log holds nothing to compare; the capture prints the results the script computes and saves their posterior means and 5% and 95% quantiles to `golden_capture.mat`, and the legacy code runs unchanged before it. Each patch is the legacy file byte for byte, with a header comment above it and the capture below it, in the file's own line endings |

Scripts with an in-script model selector (`main_UCMA.m`, `main_inflation_RV.m` and the three
`main_forecasting.m`) are captured at the selection they ship with; the other models are not
captured (decided 19 September).
