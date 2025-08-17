# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
ExMRD (Explainable Micro-Video Rumor Detection) is a Chain-of-Thought reasoning system for detecting fake news in short videos, published at WWW 2025. It combines multimodal analysis (visual, textual, contextual) using deep learning techniques.

## Core Technologies
- **Python 3.12** with PyTorch 2.6.0
- **Chinese-CLIP** (OFA-Sys/chinese-clip-vit-large-patch14) for text encoding
- **Hydra/OmegaConf** for configuration management
- **OpenAI API** required for Chain-of-Thought preprocessing

## Essential Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create --name ExMRD python=3.12
conda activate ExMRD
pip install -r requirements.txt
```

### Data Preprocessing
```bash
# Video preprocessing pipeline
bash run/preprocess.sh

# Chain-of-Thought preprocessing (requires OpenAI API)
cp .env.example .env  # Add your OpenAI API key
bash run/cot.sh
```

### Model Training
```bash
# Train on different datasets
python src/main.py --config-name ExMRD_FakeSV
python src/main.py --config-name ExMRD_FakeTT
python src/main.py --config-name ExMRD_FVC
```

## Architecture Overview

### Key Components
1. **Multimodal Feature Extraction** (`preprocess/`):
   - Frame extraction (16 frames/video)
   - OCR text extraction from frames
   - Audio transcription via Whisper
   - CLIP visual features generation

2. **Chain-of-Thought Processing** (`preprocess/cot/`):
   - **R³CoT Reasoning**: Generates reasoning chains using GPT-4
   - **Knowledge Retrieval**: External knowledge integration
   - **Textual Refining**: Corrects OCR/transcription errors
   - **Visual Refining**: Analyzes visual content

3. **ExMRD Model** (`src/model/ExMRD.py`):
   - Combines all modalities with temporal encoding
   - Implements selective layer freezing (4 layers)
   - Uses learnable positional encoding for video understanding

### Data Flow
1. Raw videos → Preprocessing → Multimodal features
2. Features + CoT reasoning → ExMRD model
3. Model outputs explainable rumor detection predictions

## Configuration System
- Training configs in `src/config/` (YAML files)
- Key parameters: batch_size=32, lr=5e-5, epochs=15
- Supports "temporal" and "5-fold" cross-validation splits
- Early stopping with patience=5

## Important Notes
- Raw video data must be obtained from original sources (only IDs provided)
- OpenAI API key required for CoT preprocessing (set in `.env`)
- GPU recommended for training (CUDA support built-in)
- Logging via Loguru and Weights & Biases integration


# ExMRD Model Q&A Documentation

## Q1: How are visual features structured before entering the SLM?

**Input**: Video features of shape `[batch_size, 16, 1024]`
- 16 frames per video
- Each frame: 1024-dimensional ViT features

**Processing**:
1. Temporal positional encoding added
2. Linear projection to 256 dimensions
3. Mean pooling across frames → `[batch_size, 256]`

## Q2: What is the SLM architecture?

The "SLM" is a **single linear layer**: `nn.LazyLinear(2)`
- Input: 256-dimensional combined features
- Output: 2 logits (binary classification)
- Not a complex language model, just a classifier

## Q3: How are text features processed?

**Text Encoder**: Chinese-CLIP BERT (`OFA-Sys/chinese-clip-vit-large-patch14`)
- 12 transformer layers (layers 4-11 fine-tuned, 0-3 frozen)
- Processes 4 CoT components independently:
  1. **OCR text** (title + refined OCR) → `data/FakeSV/CoT/{lm}/lm_text_refine.jsonl`
  2. **Visual captions** → `data/FakeSV/CoT/{lm}/lm_visual_refine.jsonl`
  3. **Commonsense knowledge** → `data/FakeSV/CoT/{lm}/lm_retrieve.jsonl`
  4. **Causal reasoning** → `data/FakeSV/CoT/{lm}/lm_reason.jsonl`

Each text component → BERT → 256-dim vector → averaged together

## Q4: How is temporal encoding implemented?

**LearnablePositionalEncoding** class:
```python
# Learnable parameters [16, 1024] initialized with sinusoidal patterns
PE[t,d] = sin(t/10000^(2i/d)) for even indices
PE[t,d] = cos(t/10000^(2i/d)) for odd indices

# Added to frames: frame[t] = frame[t] + PE[t]
```
- Preserves temporal order of 16 frames
- Parameters are learnable (can be updated during training)

## Q5: How are features combined before classification?

```python
# Video: [16, 1024] → [256] (via temporal encoding + mean)
# Text: 4×[256] → [256] (4 CoT components averaged)
#   - OCR refinement: [256] from lm_text_refine.jsonl
#   - Visual caption: [256] from lm_visual_refine.jsonl
#   - Commonsense retrieval: [256] from lm_retrieve.jsonl
#   - Causal reasoning: [256] from lm_reason.jsonl
# Final: average(video_features, text_features) → [256]
```

## Q6: What are the label mappings for FakeSV dataset?

- '真' (real) → 0
- '假' (fake) → 1  
- '辟谣' (debunking) → 0 (when included) or filtered out

## Q7: What hyperparameters control the model?

- `hid_dim`: 256 (hidden dimension)
- `dropout`: 0.3
- `num_frozen_layers`: 4 (BERT layers 0-3 frozen)
- `batch_size`: 32
- `learning_rate`: 5e-5
- `num_epoch`: 15
- `patience`: 5 (early stopping)

## Q8: How does the ablation study work?

When `ablation_no_cot=True`:
- Removes all CoT text features
- Uses only video features for classification
- Helps measure CoT contribution to performance

## Q9: What metrics are tracked?

**Macro metrics**: Accuracy, F1, Precision, Recall (averaged across classes)
**Per-class metrics**: F1, Precision, Recall for each label (Real/Fake)

## Q10: How is data split for experiments?

- **5-fold cross-validation**: Standard k-fold splits
- **Temporal split**: Train/Valid/Test split by time periods
  - Ensures model generalizes to future data

## Command Line Usage

### Run with different configurations:
```bash
# Include 辟谣 labels
python src/main.py data.include_piyao=true

# Run ablation study (no CoT)
python src/main.py data.ablation_no_cot=true

# Run on specific GPU
CUDA_VISIBLE_DEVICES=1 python src/main.py
```

### Check label distribution:
The model prints label counts at training start to verify data balance.

## Performance Notes

- Without '辟谣': ~3,624 balanced samples
- With '辟谣': ~5,495 samples (2:1 imbalanced toward real)
- Random seed (2024) ensures reproducibility
- Data cleaning doesn't affect random state