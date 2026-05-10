# Learning to Look Like Humans: Gaze-Aligned Cycling Safety Prediction

## Description

This repository contains the code for **Eye-Tracking-Guided Perceived Cycling Safety (EG-PCS)**, a pairwise learning framework for predicting perceived cycling safety from street-view imagery.

Cycling has clear public-health and environmental benefits, but many people avoid cycling when urban environments feel unsafe. Perceived safety is therefore a central barrier to cycling adoption. This project studies perceived cycling safety from pairwise comparisons of street-view images and extends existing vision-based ranking pipelines by incorporating human eye-tracking data.

The core idea is simple: a model should not only predict which street scene appears safer, but should also learn to attend to visual regions in a way that better reflects human fixation behavior. EG-PCS uses vision-transformer backbones, pairwise ranking/classification heads, and optional gaze supervision to align model attention maps with eye-tracking heatmaps.

## Architecture

<p align="center">
  <img src="docs/PCS-Net_arch.png" alt="EG-PCS architecture" width="900">
</p>

EG-PCS compares two street-view images through shared visual encoders and task-specific branches:

- **Ranking branch:** predicts which image in a pair is perceived as safer.
- **Classification branch:** supports safety-class prediction from the learned visual representation.
- **Attention branch:** introduces the main innovation of the framework by using eye-tracking heatmaps to guide or evaluate where the model attends. This branch encourages the model's visual attention to become more human-aligned, improving interpretability while preserving the pairwise safety-prediction objective.

## Dataset

We introduce the **EG-PCS dataset**, a research collection for perceived cycling safety that includes pairwise street-view comparisons, safety labels, and fixation-based gaze maps derived from eye-tracking experiments. The dataset was formed from 249 survey responses, including 26 surveys collected with eye-tracking technology.

The dataset is available for research use through Zenodo:

https://doi.org/10.5281/zenodo.20101496

Dataset documentation is provided in [docs/dataset](docs/dataset/), including a dataset card, data dictionary, license notice, and supporting dataset files.

## How to Use

Install the main dependencies:

```bash
pip install torch torchvision timm pandas numpy pyarrow scikit-learn pillow pytorch-ignite wandb optuna
```

Prepare the expected inputs:

- A pairwise comparison table, such as `comparisons_df.pickle`.
- An image directory, such as `images/`.
- Optional gaze maps for attention-guided experiments.

Train a baseline pairwise ranking model:

```bash
python train.py \
  --model rcnn \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --max_epochs 20 \
  --batch_size 32 \
  --cuda true
```

Train the gaze-aligned model:

```bash
python train.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --gaze_mode align \
  --attn_w 1.0 \
  --gaze_root Eyetracker_attention_maps \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --cuda true
```

Evaluate a trained checkpoint:

```bash
python test.py \
  --model rsscnn \
  --backbone dinov3_vitb16 \
  --comparisons comparisons_df.pickle \
  --dataset images/ \
  --resume_checkpoint path/to/checkpoint.pt \
  --cuda true
```

Useful options include `--model`, `--backbone`, `--gaze_mode`, `--attn_w`, `--batch_size`, `--max_epochs`, `--log_wandb`, and `--early_stop`.

## How to Cite

If you use this repository or build on EG-PCS, please cite the paper below. If you use the EG-PCS dataset, citation metadata for the dataset DOI is also available in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{perdigao2026learning,
  title     = {Learning to Look Like Humans: Gaze-Aligned Cycling Safety Prediction},
  author    = {Perdig{\~a}o, Lu{\'i}s Maria and Costa, Miguel and Santiago, Carlos and Marques, Manuel},
  booktitle = {Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026},
  abstract  = {Cycling delivers significant public-health and environmental benefits, yet its uptake in cities is often limited by perceived safety. When street environments appear unsafe, individuals are less likely to cycle, making perception a key barrier to adoption. Recent work has shown that pairwise comparisons of street-view images provide a scalable way to learn subjective safety judgments. However, existing approaches do not explicitly model human visual attention, which plays a central role in how humans perceive safety. We propose an Eye-Tracking-Guided Perceived Cycling Safety framework (EG-PCS) that integrates gaze data into a pairwise learning pipeline based on vision transformers. By supervising the model's attention mechanism with eye-tracking signals, we encourage alignment between learned attention maps and human fixation patterns. Experiments show that gaze-guided models achieve similar ranking performance compared to state-of-the-art approaches while producing attention maps that more accurately reflect human visual attention behavior. Our results demonstrate that incorporating eye-tracking information enhances both predictive accuracy and interpretability in perception-based urban analytics.}
}
```
