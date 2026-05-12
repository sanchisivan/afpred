# AFPRED Model Card

## Intended Use

AFPRED estimates whether short peptide sequences are likely to show antifungal activity. It is designed for exploratory screening, candidate triage, and reproducible web/API demonstrations associated with a trained sequence model.

## Inputs

- Peptide sequences of 6-40 amino acids.
- Only canonical residues are supported: `ACDEFGHIKLMNPQRSTVWY`.
- Inputs may be submitted as a single sequence, a batch, FASTA records, or CSV/TSV rows with candidate IDs.

## Outputs

- Antifungal probability score from the trained Keras model.
- Threshold-based class label.
- Peptide descriptors for interpretation and prioritization.
- Standalone peptide utility calculations when the user wants formula, mass, pI, charge, hydrophobicity, HeliQuest-like helix descriptors, Boman-like index, extinction coefficient, and liability flags without invoking the antifungal model.
- Input QC messages for unsupported residue symbols, duplicate candidates, and notation that is outside the model representation.
- Descriptor-based flags that help prioritize follow-up checks such as hemolysis, solubility, membrane activity, and sequence complexity.
- On-demand exploratory in silico variant panels that reuse the same model and descriptors to compare simple substitutions or truncations against a parent sequence.
- Property-only utility API for descriptor calculation without invoking the antifungal model.
- Downloadable report packs with CSV/FASTA data, editable SVG plots, and self-contained HTML summaries for prediction, utility, and variant workflows; utility packs include descriptor maps, aggregation-risk plots, liability summaries, and residue-composition heatmaps.
- Position-level mutation suggestions that report the best predicted single-residue substitution for each position of the selected parent peptide.

## Current Decision Labels

- `Likely antifungal`: probability greater than or equal to the selected threshold.
- `Intermediate / review`: probability from 0.50 up to the selected threshold.
- `Low antifungal signal`: probability below 0.50.

The default threshold is 0.80 and should be justified in the manuscript using validation metrics, calibration behavior, or a target operating point.

## Limitations

- The app does not establish biological activity by itself.
- The model may not generalize to peptide classes underrepresented in training.
- Non-canonical amino acids, post-translational modifications, D-amino acids, cyclization, terminal modifications, and formulation effects are not represented.
- The descriptors are approximate and are not substitutes for specialized toxicity, hemolysis, solubility, stability, or structure prediction workflows.
- Descriptor flags are interpretive aids, not additional experimentally trained endpoints.
- Utility-tab calculations assume linear canonical sequences and do not yet represent modified residues, salts, cyclization, D-amino acids, or protecting groups.
- Variant and mutation suggestions are computational what-if analyses and should not be interpreted as optimized sequences without experimental testing.
- For batch screens, mutational analysis should be run only after ranking and selecting a manageable set of parent peptides.

## Recommended Paper Reporting

- Dataset origin, curation rules, and duplicate/redundancy filtering.
- Definition of antifungal and non-antifungal classes.
- Train/validation/test split strategy, including sequence identity controls.
- Model architecture and preprocessing details.
- Primary metrics: ROC-AUC, PR-AUC, sensitivity, specificity, MCC, F1, and calibration.
- Threshold selection rationale.
- External validation, if available.
- Web server URL, source repository, model version, and reproducibility instructions.
