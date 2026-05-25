# LLM-Group-Assignment
Assignments for the "Transformers and Large Language Models" Course.
This project is based on https://github.com/KRLabsOrg/LettuceDetect.

### Installation

Install from the repository:
```bash
pip install -e .
```

### Training a Model

You can train the model with the following command.

```bash
python scripts/train.py \
    --ragtruth-path ./Dataset/type1_evident_conflict.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/hallucination_detector \
    --batch-size 8 \
    --epochs 8 \
    --learning-rate 5e-5 
```

```bash
python scripts/train_lora.py \
    --ragtruth-path ./Dataset/type2_baseless_info.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/hallucination_detector \
    --batch-size 8 \
    --epochs 5 \
    --learning-rate 2e-5 
```

```bash
python scripts/train_lora.py \
    --ragtruth-path ./Dataset/type3_missing_tool.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/hallucination_detector \
    --batch-size 8 \
    --epochs 5 \
    --learning-rate 5e-5 
```
