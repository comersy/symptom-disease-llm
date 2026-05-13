import os
import json
import re
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "models/mistral-7b-instruct-v0.3"
LORA_WEIGHTS = "outputs/mistral-disease-lora/final"
DATA_DIR     = "data/dataset"
LABEL_MAP    = "data/label_map.json"
MAX_NEW_TOKENS = 20
BATCH_SIZE   = 8
# ─────────────────────────────────────────────────────────────────────────────


def load_model(base_model: str, lora_path: str):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batch generation

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


def main():
    # 1. Load test dataset
    ds      = load_from_disk(DATA_DIR)
    test_ds = ds["test"]
    labels  = test_ds["label"]

    # Strip the answer part from prompts for inference
    prompts = [ex["text"].split("[/INST]")[0] + "[/INST]" for ex in test_ds]
    print(f"[INFO] Evaluating on {len(prompts)} examples ...")

    # 2. Load model
    model, tokenizer = load_model(BASE_MODEL, LORA_WEIGHTS)

    # 3. Generate predictions in batches
    all_preds = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Inference"):
        batch = prompts[i : i + BATCH_SIZE]
        all_preds.extend(generate_predictions(model, tokenizer, batch))

    # 4. Fuzzy match predictions against known labels
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

    # 6. Show first errors
    errors = [(p, m, r) for p, m, r in zip(prompts, matched_preds, labels) if m != r]
    print(f"\n[INFO] {len(errors)} errors out of {len(prompts)}\n")
    for prompt, pred, real in errors[:5]:
        snippet = prompt.split("Patient symptoms:\n")[-1][:120].replace("\n", " ")
        print(f"  Real    : {real}")
        print(f"  Predicted: {pred}")
        print(f"  Symptoms : {snippet}...\n")

    # 7. Save results
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "label": labels,
        "predicted": matched_preds,
        "raw_pred": all_preds
    }).to_csv("outputs/evaluation_results.csv", index=False)
    print("Results saved to outputs/evaluation_results.csv")


if __name__ == "__main__":
    main()