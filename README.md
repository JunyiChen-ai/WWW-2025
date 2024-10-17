# ExMRD

This repo provides a official implementation of ExMRD as described in the paper:

**Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning**


## Source Code Structure

```
data        # dir of each dataset
- FakeSV
- FakeTT
- FVC

preprocess  # code to MLLM CoT process

src
- config    # training config
- model     # ExMRD model
- utils     # training utils
- data      # dataloader of ExMRD
```

## Dataset

We provide video IDs for each dataset in both temporal and five-fold splits. Due to copyright restrictions, the raw datasets are not included. You can obtain the datasets from their respective original project sites.

### FakeSV

Access the full dataset from[ICTMCG/FakeSV: Official repository for "FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms", AAAI 2023.](https://github.com/ICTMCG/FakeSV).

### FakeTT

Access the full dataset from [ICTMCG/FakingRecipe: Official Repository for "FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process", ACM MM 2024](https://github.com/ICTMCG/FakingRecipe).

### FVC

Access the full dataset from [MKLab-ITI/fake-video-corpus: A dataset of debunked and verified user-generated videos.](https://github.com/MKLab-ITI/fake-video-corpus).

# Usage

## Requirements

To set up the environment, run the following commands:

```bash
conda create --name py312 python=3.12
pip install torch transformers tqdm loguru pandas torchmetrics scikit-learn colorama wandb hydra-core
```

## Data Preprocess

1. Sample 16 frames from each video in the dataset and store them in `{dataset}/frames_16`.

2. Extract on-screen text from keyframes using Paddle-OCR and save to `{dataset}/ocr.jsonl`.

3. Extract audio transcripts from video audio using Whisper and save to `{dataset}/transcript.jsonl`.

4. Extract visual features from each video using a pre-trained CLIP-ViT model and save to `{dataset}/fea/vit_tensor.pt`.

## MLLM R^3CoT Process

```bash
# Run textual refining
python preprocess/run_textual_refining.py

# Run visual refining
python preprocess/run_visual_refining.py

# Run retrieving
python preprocess/run_retrieving.py

# Run reasoning
python preprocess/run_reasoning.py
```

## Run

```bash
# Run ExMRD for the FakeSV dataset
python src/main.py --config-name ExMRD_FakeSV

# Run ExMRD for the FakeTT dataset
python src/main.py --config-name ExMRD_FakeTT

# Run ExMRD for the FVC dataset
python src/main.py --config-name ExMRD_FVC
```