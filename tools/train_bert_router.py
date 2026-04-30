"""
BERT 路由器训练脚本（LoRA 微调，保护中文能力）。

使用 router_train_cn.jsonl 全量训练二分类器（0=保持 flash, 1=升格 pro），
LoRA 注入 attention Q/V（仅 2 模块，避免灾难性遗忘），全量数据训练，不切验证集。

测试集为单独准备的 router_test_cn.jsonl。

用法:
    python tools/train_bert_router.py --epochs 3
    python tools/train_bert_router.py --max-samples 50000 --epochs 3
    python tools/train_bert_router.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import warnings
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# 消除 BERT 内部 use_return_dict 弃用告警
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# LoRA 配置（保护中文预训练能力，仅注入 attention Q/V）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]


# ── 数据加载 ────────────────────────────────────────────────────────────

def load_dataset(
    jsonl_path: Path,
    max_samples: int | None = None,
    shuffle_seed: int = RANDOM_SEED,
) -> Dataset:
    """从 JSONL 加载数据，取 text 和 label 字段。"""
    texts, labels = [], []
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    if max_samples and max_samples < len(lines):
        rng = random.Random(shuffle_seed)
        rng.shuffle(lines)
        lines = lines[:max_samples]

    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        texts.append(item["text"])
        labels.append(item["label"])

    logger.info(
        "加载 %d 条数据（label=0: %d, label=1: %d）",
        len(texts), labels.count(0), labels.count(1),
    )
    return Dataset.from_dict({"text": texts, "label": labels})


# ── 分词（静态 padding，一次性 CPU 预处理） ────────────────────────────

def tokenize_fn(examples: dict, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


# ── 主流程 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="训练 BERT 升格路由器（LoRA 微调）",
    )
    parser.add_argument(
        "--model-path",
        default="base_model",
        help="预训练模型目录或 HF model ID",
    )
    parser.add_argument(
        "--data",
        default="datasets/router_train_cn.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--output",
        default="router_model",
        help="模型保存目录（默认 router_model/）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大训练样本数（None=全部 375K）",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="训练轮数（默认 3）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="训练 batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="学习率（LoRA 大学习率默认 1e-3）"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="最大 token 长度"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2000,
        help="保存 checkpoint 步数",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="禁用 FP16 混合精度训练",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_path = Path(args.data)
    model_path = Path(args.model_path)
    output_dir = Path(args.output)

    # ── 设备检测 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda" and not args.no_fp16
    logger.info("设备: %s, FP16: %s", device, use_fp16)

    # ── 加载数据 ──
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    logger.info("加载数据: %s", data_path)
    ds = load_dataset(data_path, max_samples=args.max_samples)

    # ── 加载模型 + LoRA 注入 ──
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}\n"
            f"请先下载 BERT 基础模型到指定目录。"
        )

    logger.info("加载模型 + tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=2,
        return_dict=True,
        ignore_mismatched_sizes=True,
    )

    logger.info(
        "注入 LoRA (r=%d, alpha=%d, dropout=%.2f, targets=%s)",
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "LoRA: %s / %s 参数可训练 (%.1f%%)",
        f"{trainable_params:,}", f"{total_params:,}",
        trainable_params / total_params * 100,
    )

    # ── 分词（静态 padding） ──
    logger.info("分词 (max_length=%d)...", args.max_length)
    ds = ds.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    logger.info("全量训练 %d 条（不切验证集，测试集单独提供）", len(ds))

    # warmup steps
    total_steps = (len(ds) // args.batch_size) * args.epochs
    warmup_steps = max(100, int(total_steps * 0.06))

    # ── 训练参数（纯训练，无 eval/验证逻辑） ──
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=200,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=use_fp16,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=RANDOM_SEED,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    logger.info(
        "开始训练 (%d epochs, batch=%d, lr=%.0e, warmup=%d, ~%d steps)...",
        args.epochs,
        args.batch_size,
        args.lr,
        warmup_steps,
        total_steps,
    )
    trainer.train()

    # ── 保存模型（merge LoRA → 纯 HuggingFace 格式，tokenizer 不变） ──
    logger.info("合并 LoRA 并保存到: %s", output_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(
        "训练完成！请在 config.yaml 中设置:\n"
        "  router_type: bert\n"
        "  bert_checkpoint: \"%s\"",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
