# AFPRED Antifungal Peptide Predictor

AFPRED is a Flask web application for screening short peptide sequences with a trained Keras model for antifungal activity. It accepts individual peptides, plain-text batches, or FASTA-formatted records and returns model scores together with peptide descriptors useful for candidate prioritization.

## Current Features

- Single and batch prediction for peptides of 6-40 canonical amino acids.
- FASTA, one-sequence-per-line, comma/semicolon separated, CSV, or TSV input with candidate IDs preserved when an ID/name column is supplied.
- Adjustable decision threshold.
- CSV export of predictions and descriptors plus clean FASTA export for valid sequences.
- JSON API endpoints for prediction, variant exploration, and property-only utilities.
- On-demand variant exploration endpoint for activity-boosting substitution scans, alanine scans, tryptophan scans, charge scans, hydrophobicity tempering, and terminal truncations.
- Health endpoint for deployment monitoring.
- Input QC for unsupported residues, terminal-modification text, duplicate sequences, and batch-level invalid records.
- Peptide descriptors: length, molecular formula, approximate molecular weight, net charge at pH 7 and pH 5.5/7.4, charge density, estimated isoelectric point, GRAVY hydropathy, hydrophobic moment, aliphatic index, extinction coefficient, hydrophobic/positive/negative/polar residue percentage, aromaticity, cysteine count, sequence entropy, terminal profile, sliding-window profiles, composition, and profile tags.
- Interpretation flags for membrane-active profiles, high hydrophobicity, low cationic character, aromatic enrichment, cysteine-rich candidates, basic residue clusters, and low-complexity sequences.
- Descriptor-based screening tier to separate candidates to advance, review, or deprioritize.
- Position-level mutation summary for the best predicted single-residue substitution at each position of the selected parent peptide.
- Variant CSV and FASTA exports with parent sequence, mutation, score delta, rationale, and core descriptor fields.
- A standalone Utilities tab for PepCalc-style calculations without running the antifungal model.
- A tabbed scientific workbench interface with prediction, design, utilities, batch/export, model-readiness, and information/contact sections.
- External model integration notes for AMP, toxicity, hemolysis, and antifungal-protein predictors that could be wired in as optional screening backends.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`.

The app expects the trained model at:

```text
antifungal_peptide_model.h5
```

## Web Input

Plain batch input:

```text
KWKLFKKILKFLHSAKKF
GLFDIIKKIAESF
```

FASTA input:

```text
>candidate_01
KWKLFKKILKFLHSAKKF
>candidate_02
GLFDIIKKIAESF
```

CSV/TSV input with IDs:

```text
id,sequence
candidate_01,KWKLFKKILKFLHSAKKF
candidate_02,GLFDIIKKIAESF
```

Only the 20 canonical amino acids are accepted. Invalid sequences are reported in the results instead of stopping the whole batch.

## JSON API

Endpoint:

```text
POST /api/predict
```

Example body:

```json
{
  "threshold": 0.8,
  "sequences": [
    "KWKLFKKILKFLHSAKKF",
    "GLFDIIKKIAESF"
  ]
}
```

The response includes a batch summary and per-peptide prediction rows with sequence QC, descriptors, composition, duplicate status, ranking, and interpretation flags.

Variant exploration endpoint:

```text
POST /api/variants
```

Example body:

```json
{
  "threshold": 0.8,
  "mode": "alanine_scan",
  "sequence": "KWKLFKKILKFLHSAKKF"
}
```

Supported modes are `activity_optimization`, `alanine_scan`, `tryptophan_scan`, `lysine_scan`, `hydrophobic_tempering`, and `terminal_truncation`. Design utilities are not run automatically for large batches; first rank the library, then run a selected mutational analysis for the top-scoring valid parent.

Property-only endpoint:

```text
POST /api/properties
```

Example body:

```json
{
  "sequences": [
    "KWKLFKKILKFLHSAKKF",
    "GLFDIIKKIAESF"
  ]
}
```

This endpoint returns the Utilities-tab descriptors without loading or invoking the antifungal model.

## Utilities Tab

The Utilities tab computes physicochemical properties without invoking the AFPRED model. It accepts the same plain-text, CSV/TSV-like, or FASTA inputs and exports a utility CSV with formula, mass, pI, pH-dependent charge, GRAVY/KD and Eisenberg hydrophobicity descriptors, hydrophobic moment, hydrophobic face angle, HeliQuest-like discriminant, Boman-like index, disorder-promoting fraction, aggregation-risk descriptors, extinction coefficients, and sequence-level liability flags for oxidation, deamidation/isomerization, cyclization, protease-like cleavage motifs, solubility, aggregation, and charge/pI context.

## Deployment

The included Render configuration installs dependencies from `requirements.txt` and starts the app with:

```bash
python app.py
```

## Model Scope

This application is intended as a screening and prioritization aid. Predictions should be interpreted together with model validation results, peptide physicochemical properties, novelty checks, toxicity/hemolysis assessment, and experimental validation.

For publication, report the training dataset source, positive/negative class definition, redundancy filtering, train/validation/test split strategy, model architecture, calibration, threshold selection, and external validation performance.
