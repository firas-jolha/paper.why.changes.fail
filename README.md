# Replication package

## Data
ApacheJIT (Keshavarz & Nagappan, MSR 2022). Place `apachejit_total.csv` in `data/`. The file used here was taken from a public fork of the authors' repository and verified against `data/apache_metrics_kamei.csv` in https://github.com/hosseinkshvrz/apachejit (all 106,674 commits match). Edit the path in `scripts/common.py` if needed.


## Steps
1. `python scripts/a1_main.py` – time-aware split (train 2003–2016, test 2017–2019), LR and gradient boosting, grouped and per-metric permutation importance, Spearman correlations -> `results/main.json`
2. `python scripts/a2_time_project.py` – per test year (3-year training windows) and per project (70/30) -> `results/time_project.json`
3. `python scripts/a3_conditional_stability.py` – conditional importance within size x FIX strata; Kendall's W across 20 bootstraps, 11 years and 14 projects -> `results/cond_stability.json`
4. `python scripts/a4_figures.py` – figures and share tables.


## Cross-check on ReDef (revert-based labels)
Data: ReDef (Nam et al., FSE 2026), released by its authors (figshare link in the ReDef paper).
Place `1_train.jsonl`, `1_valid.jsonl`, `1_test.jsonl` in `data/redef/`. Verified totals: 13,432 changes, 3,164 defective, 22 projects.
The GPT-produced fields `commit_type` and `explanation` are NOT used (they encode the label).
5. `python scripts_redef/load.py`
6. `python scripts_redef/features.py` – lines added/deleted, hunks, function length, fix keyword, weekday
7. `python scripts_redef/models.py` – LR and gradient boosting, grouped permutation importance, single-feature AUCs, bootstrap CI
8. `python scripts_redef/apachejit_control.py` – size AUC on single-file / small ApacheJIT commits
9. `python scripts_redef/fig_tables.py` – Table 5 and the two-dataset figure (needs results/main.json and results/prauc.json; PR-AUC for ApacheJIT is computed with the a1 setup)