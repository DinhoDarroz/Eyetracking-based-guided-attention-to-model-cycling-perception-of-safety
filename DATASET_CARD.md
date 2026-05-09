# EG-PCS Dataset Card

## Dataset Summary

EG-PCS contains pairwise perceived cycling safety comparisons, street-view image
pairs, and fixation-based gaze maps derived from an eye-tracking experiment. It
supports research on perceived urban safety, pairwise visual ranking, gaze-guided
learning, and attention alignment.

## Data Instances

Each row in `comparisons/comparisons.csv` represents one pairwise comparison
between a left image and a right image. The row contains the ground-truth
pairwise label, image references, optional gaze-map references, and anonymized
survey/trial metadata.

## Labels

The `score` field is the ground-truth pairwise label:

- `-1`: left image is perceived as safer.
- `0`: tie.
- `+1`: right image is perceived as safer.

## Gaze Maps

Gaze maps are stored as NumPy `.npy` arrays and were generated from fixation data
with `survey_eye_tracker/build_fixation_based_attention_maps_ogama_like.py`.
They are released as derived attention maps rather than raw participant gaze
streams.

## Intended Uses

- Pairwise perceived cycling safety prediction.
- Gaze-guided computer vision and attention-alignment experiments.
- Human visual attention analysis for urban imagery.
- Reproducibility of EG-PCS experiments.

## Out-of-Scope Uses

The dataset should not be used to identify individual study participants or to
make high-stakes decisions about individual people. It should not be interpreted
as a complete, universal measure of urban safety.

## Privacy and Ethics

The public release should include only anonymized survey/trial metadata and
derived gaze maps. Do not include raw personally identifying eye-tracking records
unless the relevant consent and ethics approvals explicitly allow it.

## Known Limitations

The labels reflect perceived safety judgments in the survey setting and may be
affected by participant demographics, image source coverage, city selection, and
street-view capture conditions. Gaze maps represent visual attention during the
task, not causal explanations of safety perception.

