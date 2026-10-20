# Multi-omics drug sensitivity prediction in cancer cell lines

Analysis code for "Transcriptomic Profiles Carry Most of the Multi-Omics Signal for Drug Sensitivity Prediction in
Cancer Cell Lines: A Cross-Lineage and Independent-Screen Evaluation of Twelve Anticancer Agents".

Twelve anticancer drugs, 509 cell lines, DepMap expression, mutation and copy number data, GDSC2 response for
training and PRISM response for external testing. Models are XGBoost classifiers with fixed hyperparameters; all
feature selection, imputation and scaling are fitted inside training folds. All random seeds are fixed.

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Place the inputs as described in `data/README.md`, or point to them with environment variables:
`DATA_DIR`, `RAW_DIR`, `PUB_DIR`, `OUT_DIR` (see `scripts/common.py`).

## Scripts (run in order from the repository root)
| Script | Analysis | Main outputs (in `results/`) |
|---|---|---|
| `scripts/01_cross_validation.py` | 3 x 5-fold stratified CV with fold-level AUCs: tissue, OncoTree subtype, expression, curated-biomarker expression, early fusion, curated early fusion, late fusion; median, tertile-extreme and lower-quartile response definitions | `cv/*.npz`, `cv/*_folds.csv` |
| `scripts/02_summary_statistics.py` | random-effects pooled fusion gain, equivalence bounds, minimum detectable differences; per-repeat DeLong tests; paired bootstrap of regression gains; concordance robustness (bootstrap, leave-one-drug-out, partial correlation); LOCO uncertainty and minimum lineage size; copy-number missingness | `a_*.csv` to `e_*.csv`, `summary_stats.json` |
| `scripts/03_external_validation_prism.py` | external validation on PRISM with three sensitivity thresholds and a threshold-free Spearman correlation | `external_validation.csv`, `external/*.npz` |
| `scripts/04_cn_expression_redundancy.py` | copy number versus expression of the same gene | `cn_redundancy.json` |
| `scripts/05_cv_summary.py` | DeLong and Nadeau-Bengio corrected resampled t-tests with Holm adjustment on the output of `01`; curated-biomarker, subtype and alternative-threshold comparisons; ranking stability | `cv_performance.csv`, `cv_comparisons.csv`, `cv_summary.json` |

`01` takes about 70 minutes on 12 cores; `03` about 30 minutes; the others run in minutes.

`scripts/common.py` holds the configuration, data loading, fold-internal preprocessing, curated biomarker lists,
model settings, the fast DeLong implementation and the corrected resampled t-test.

## Supporting tables
The drug-cell line inclusion table (Table S21) and the features selected for each drug (Table S22) are provided with
the article's Supplementary Materials.

## Citation
Please cite the article (details to be added on publication).
