# EG-PCS Dataset Card

## Summary

EG-PCS contains pairwise perceived cycling safety comparisons, street-view image pairs, and fixation-based gaze maps from an eye-tracking experiment. The dataset supports research on subjective urban safety, pairwise visual ranking, gaze-guided learning, and human-aligned attention models.

## Dataset Formation

The dataset was formed from 249 survey responses, including 26 surveys collected with eye-tracking technology. Participants evaluated perceived cycling accident safety across different street-view environments using pairwise image comparisons. The eye-tracking subset provides fixation-derived gaze annotations for 1,419 comparisons.

<p align="center">
  <img src="../example_trial.png" alt="Example eye-tracking survey trial with gaze maps overlaid on the two compared images" width="800">
</p>

The example above shows one eye-tracking trial used as part of the dataset foundation, with gaze maps overlaid on the two compared street-view images.

## Composition

| Dataset subset | y=-1 | y=0 | y=1 | Total | Gaze subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barcelona | 389 | 334 | 430 | 1,153 | -- |
| Berlin | 2,905 | 1,363 | 3,002 | 7,270 | 999 |
| London UK Collideoscope | 204 | 171 | 184 | 559 | -- |
| London UK Gov | 184 | 184 | 191 | 559 | -- |
| Munich | 198 | 107 | 228 | 533 | -- |
| Paris | 176 | 179 | 194 | 549 | -- |
| Sequences | 627 | 1,487 | 886 | 3,000 | 420 |
| **Total** | **4,683** | **3,825** | **5,115** | **13,623** | **1,419** |

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

The underlying survey was approved by Instituto Superior Técnico’s Ethics Committee and captured perceived cycling accident safety in different environments using pairwise image comparisons. The public dataset uses anonymized survey/trial metadata and derived gaze maps. It is not intended for identifying individual study participants or for making high-stakes decisions about individual people.
