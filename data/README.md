# Data layout

All inputs are public. Nothing in this folder is tracked by git.

```
data/
  raw/                 downloads (only needed for 03_external_validation_prism.py)
    PRISM_secondary_dose_response.csv     PRISM Repurposing secondary screen, release 20Q2
  processed/           matched matrices and outputs of the primary analysis
    expr.pkl  cn.pkl  mut.pkl  lineage.pkl  labels.csv
    summary_*.csv  cv/*.npz
  published_tables/    Supplementary Tables S1-S7 of the article (CSV)
```

Sources
- DepMap release 26Q1 (https://depmap.org/portal): Model.csv, OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv,
  OmicsSomaticMutations.csv, OmicsCNGeneWGS.csv
- GDSC2 fitted dose response, release 27 October 2023, Cell Model Passports (https://cellmodelpassports.sanger.ac.uk)
- PRISM Repurposing secondary screen 20Q2 (https://depmap.org/repurposing)

`expr.pkl`, `cn.pkl` and `mut.pkl` are cell line x gene matrices (rows indexed by DepMap ModelID) for the 1105 models
with all three omics layers; `labels.csv` holds LN_IC50 per drug and cell line with OncoTree lineage and primary
disease. The pickles were written with pandas 3 and need pandas >= 3 to load.
