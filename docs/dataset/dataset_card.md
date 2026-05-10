# EG-PCS Dataset Card

## Summary

EG-PCS contains pairwise perceived cycling safety comparisons, street-view image pairs, and fixation-based gaze maps from an eye-tracking experiment. The dataset supports research on subjective urban safety, pairwise visual ranking, gaze-guided learning, and human-aligned attention models.

## Data Instances

Each row in the comparison table represents one pairwise judgment between a left street-view image and a right street-view image. Rows include the perceived safety label, image references, optional gaze-map references, and anonymized survey/trial metadata.

## Labels

The `score` field is the pairwise ground-truth label:

- `-1`: the left image is perceived as safer.
- `0`: both images are perceived as similarly safe.
- `+1`: the right image is perceived as safer.

## Gaze Maps

Gaze maps are released as NumPy `.npy` arrays derived from fixation data. They represent visual attention during the perceived-safety comparison task and are intended for gaze-guided training, attention-alignment evaluation, and interpretability analysis.

## Intended Uses

- Pairwise perceived cycling safety prediction.
- Gaze-guided computer vision experiments.
- Attention-alignment and interpretability studies.
- Urban perception research using street-view imagery.

## Limitations

The labels reflect perceived safety judgments collected in a specific survey setting. They may be influenced by participant demographics, city coverage, image-source coverage, and street-view capture conditions. Gaze maps represent attention during the task and should not be interpreted as complete causal explanations of perceived safety.

## Ethics and Privacy

The public dataset uses anonymized survey/trial metadata and derived gaze maps. It is not intended for identifying individual study participants or for making high-stakes decisions about individual people.
