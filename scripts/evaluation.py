import os
import json
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "models/mistral-7b-instruct-v0.3"
LORA_WEIGHTS   = "outputs/mistral-disease-lora/final"
DATA_DIR       = "data/dataset"
LABEL_MAP      = "data/label_map.json"
MAX_NEW_TOKENS = 20
BATCH_SIZE     = 8
# ─────────────────────────────────────────────────────────────────────────────


def load_model(base_model: str, lora_path: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("models/mistral-7b-instruct-v0.3", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def extract_prediction(generated_text: str, prompt: str) -> str:
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[-1]
    else:
        response = generated_text[len(prompt):]
    response = re.sub(r"</s>.*", "", response, flags=re.DOTALL)
    return response.split("\n")[0].strip()


def generate_predictions(model, tokenizer, prompts: list) -> list:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return [extract_prediction(d, p) for d, p in zip(decoded, prompts)]


def normalize(text: str) -> str:
    return text.lower().strip()


def plot_confusion_matrix(labels, preds, classes, save_path):
    cm = confusion_matrix(labels, preds, labels=classes)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix — Mistral Disease Predictor", fontsize=15, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved to {save_path}")


def plot_precision_recall(labels, preds, classes, save_path):
    report = classification_report(labels, preds, labels=classes, output_dict=True, zero_division=0)

    precisions = [report.get(c, {}).get("precision", 0) for c in classes]
    recalls    = [report.get(c, {}).get("recall", 0) for c in classes]

    x = range(len(classes))
    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar([i - 0.2 for i in x], precisions, width=0.4, label="Precision", color="steelblue")
    bars2 = ax.bar([i + 0.2 for i in x], recalls,    width=0.4, label="Recall",    color="tomato")

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Precision & Recall per Disease — Mistral Disease Predictor", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Precision/Recall chart saved to {save_path}")


def main():
    # 1. Load test dataset
    ds      = load_from_disk(DATA_DIR)
    test_ds = ds["test"]
    labels  = list(test_ds["label"])

    prompts = [ex["text"].split("[/INST]")[0] + "[/INST]" for ex in test_ds]
    print(f"[INFO] Evaluating on {len(prompts)} examples ...")

    # 2. Load model
    model, tokenizer = load_model(BASE_MODEL, LORA_WEIGHTS)

    # 3. Generate predictions
    all_preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Inference"):
        batch = prompts[i : i + BATCH_SIZE]
        all_preds.extend(generate_predictions(model, tokenizer, batch))

    # 4. Fuzzy match
    with open(LABEL_MAP) as f:
        known_labels = list(json.load(f).keys())

    def match_label(pred: str) -> str:
        norm = normalize(pred)
        for l in known_labels:
            if normalize(l) in norm or norm in normalize(l):
                return l
        return pred

    matched_preds = [match_label(p) for p in all_preds]

    # 5. Metrics
    acc = accuracy_score(labels, matched_preds)
    print(f"\n{'─'*50}")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"{'─'*50}\n")
    print(classification_report(labels, matched_preds, zero_division=0))

    # 6. Errors
    errors = [(p, m, r) for p, m, r in zip(prompts, matched_preds, labels) if m != r]
    print(f"\n[INFO] {len(errors)} errors out of {len(prompts)}\n")
    for prompt, pred, real in errors[:5]:
        snippet = prompt.split("Patient symptoms:\n")[-1][:120].replace("\n", " ")
        print(f"  Real     : {real}")
        print(f"  Predicted: {pred}")
        print(f"  Symptoms : {snippet}...\n")

    # 7. Save CSV
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "label": labels,
        "predicted": matched_preds,
        "raw_pred": all_preds
    }).to_csv("outputs/evaluation_results.csv", index=False)
    print("Results saved to outputs/evaluation_results.csv")

    # 8. Plots
    classes = sorted(set(labels))
    plot_confusion_matrix(labels, matched_preds, classes, "outputs/confusion_matrix.png")
    plot_precision_recall(labels, matched_preds, classes, "outputs/precision_recall.png")


if __name__ == "__main__":
    main()