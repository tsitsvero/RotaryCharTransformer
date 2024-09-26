# Character-Level Language Model with Rotary Position Embeddings

This project implements and compares two character-level language models trained on the enwik8 dataset: a baseline transformer model and an enhanced version using Rotary Position Embeddings (RoPE).

## Project Overview

- **Dataset**: enwik8 (~100M characters from Wikipedia)
- **Task**: Character-level language modeling
- **Baseline**: Standard GPT-style transformer
- **Novel Approach**: GPT with Rotary Position Embeddings (RoPE)

## Table of Contents

1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Model Implementation](#model-implementation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)

## Setup

1. Clone the nanoGPT repository:
   ```
   git clone https://github.com/RiddhiRaj/RotaryCharTransformer.git

   cd RotaryCharTransformer
   ```

2. Install required packages:
   ```
   pip install transformers datasets tiktoken wandb tqdm numpy torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   ```

## Data Download & Preparation

1. Download the dataset (press "y" when asked to run custom code)
   ```python
   python download_enwik.py
   ```

2. Move the prepared data:
   ```
   mkdir -p data/enwik8
   mv train.txt valid.txt test.txt data/enwik8/
   ```

3. Load and split the enwik8 dataset:
   ```python
   python prepare_enwik8.py
   ```



## Model Implementation

- Baseline model: Implemented in `model_baseline.py`
- RoPE model: Implemented in `model_rope.py`

## Training

1. Train the baseline model:
   ```
   python train.py --config config/enwik8_char_baseline.py
   ```

2. Train the RoPE model:
   ```
   python train.py --config config/enwik8_char_rope.py
   ```

## Evaluation

Evaluate both models:
```
python evaluate.py --model_type gpt --checkpoint out-enwik8-char/ckpt.pt
python evaluate.py --model_type rope --checkpoint out-enwik8-char-rope/ckpt.pt
```

## Results

The GPTWithRoPE model significantly outperforms the baseline, demonstrating the effectiveness of Rotary Position Embeddings in character-level language modeling.

For more detailed information on the implementation and analysis, please refer to the Jupyter notebook and individual Python files in this repository.