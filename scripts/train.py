import os
import json
import torch
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk


@dataclass
class Config:
    # Model
    model_name: str = "models/mistral-7b-instruct-v0.3"
    # Data
    data_dir: str = "data/dataset"
    # Output
    output_dir: str = "outputs/mistral-disease-lora"
    # 4-bit quantization (QLoRA — reduces VRAM usage significantly)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    # LoRA — instead of updating all 7B weights, we inject small trainable matrices
    lora_r: int = 16      # rank: higher = more capacity, more VRAM
    lora_alpha: int = 32      # scaling factor (effective scale = alpha/r)
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Training
    max_seq_length: int = 512
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int  = 4
    gradient_accumulation_steps: int = 2   # effective batch size = 4 * 2 = 8
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    bf16: bool = True    # use fp16=True instead if GPU is not Ampere+
    fp16: bool = False
    # Logging & checkpointing
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    seed: int = 42
    report_to: str = "none"  # set to "wandb" to enable experiment tracking


cfg = Config()


def get_bnb_config() -> BitsAndBytesConfig:
    # Quantize base model weights to 4-bit so they fit in VRAM
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )


def get_lora_config() -> LoraConfig:
    # LoRA injects trainable rank-r matrices into the target attention/MLP layers
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )


def load_model_and_tokenizer():
    print(f"[INFO] Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token  # Mistral has no dedicated pad token

    print(f"[INFO] Loading model (4-bit={cfg.use_4bit}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=get_bnb_config() if cfg.use_4bit else None,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False       # required when using gradient checkpointing
    model.config.pretraining_tp = 1

    if cfg.use_4bit:
        # Prepares quantized model to accept gradient updates through LoRA
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()   # should show ~0.3% of total params

    return model, tokenizer


def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. Load dataset prepared by prepare_data.py
    print(f"[INFO] Loading dataset from {cfg.data_dir}")
    ds = load_from_disk(cfg.data_dir)
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    print(f"[INFO] Train={len(train_ds)}, Val={len(val_ds)}")

    # 2. Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,     # trades compute for VRAM savings
        optim="paged_adamw_32bit",       # memory-efficient optimizer for QLoRA
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        group_by_length=True,            # speeds up training by batching similar-length sequences
        report_to=cfg.report_to,
        seed=cfg.seed,
    )

    # 4. SFTTrainer handles tokenization and causal LM loss automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",       # column containing the full formatted prompt
        max_seq_length=cfg.max_seq_length,
        packing=False,                   # set True to pack short sequences for faster training
    )

    # 5. Train
    print("\n[INFO] Starting training ...")
    trainer.train()

    # 6. Save final LoRA weights + tokenizer
    final_path = os.path.join(cfg.output_dir, "final")
    print(f"\n[INFO] Saving model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    with open(os.path.join(final_path, "train_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()