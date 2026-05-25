import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model, TaskType  # 新增

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
    HallucinationSample,
)
from lettucedetect.models.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detector with LoRA")
    parser.add_argument("--ragtruth-path", type=str, default="data/ragtruth/ragtruth_data.json")
    parser.add_argument("--ragbench-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", type=str, default="output/hallucination_detector_lora")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    # LoRA 参数
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


def split_train_dev(samples, dev_ratio=0.1):
    n = len(samples)

    # 每隔 step 取一个
    step = int(1 / dev_ratio)

    dev_indices = set(range(step - 1, n, step))

    train_samples = [
        sample for i, sample in enumerate(samples)
        if i not in dev_indices
    ]

    dev_samples = [
        sample for i, sample in enumerate(samples)
        if i in dev_indices
    ]

    return train_samples, dev_samples


def main():
    set_seed(123)
    args = parse_args()

    # 数据加载（与原代码相同）
    ragtruth_path = Path(args.ragtruth_path)
    with open(ragtruth_path, "r", encoding="utf-8") as f:
        ragtruth_json = [json.loads(line) for line in f if line.strip()]
    ragtruth_data = HallucinationData.from_json(ragtruth_json)
    ragtruth_train_samples = [s for s in ragtruth_data.samples if s.split == "train"]
    train_samples, dev_samples = split_train_dev(ragtruth_train_samples)

    if args.ragbench_path:
        ragbench_path = Path(args.ragbench_path)
        ragbench_data = HallucinationData.from_json(json.loads(ragbench_path.read_text()))
        train_samples.extend([s for s in ragbench_data.samples if s.split == "train"])
        dev_samples.extend([s for s in ragbench_data.samples if s.split == "dev"])

    # 加载 tokenizer 和基础模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        trust_remote_code=True,
    )

    # ---------- LoRA 配置 ----------
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # ModernBERT 的注意力线性层命名，可根据实际模型结构调整
        target_modules=["Wqkv", "Wo", "Wi", "Wo"],  # 常见名称
        # 更通用的方式：自动查找所有线性层（谨慎使用，模块过多）
        # target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # 验证只训练 LoRA 参数
    # ---------------------------------

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataset = HallucinationDataset(train_samples, tokenizer)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )

    # 自定义 Trainer 应只更新 requires_grad=True 的参数，LoRA 模型默认如此
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
    )
    trainer.train()

    # 保存 LoRA adapter（不包含 base model）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()