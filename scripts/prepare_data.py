import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

CSV_PATH   = "Symptom2Disease.csv"
OUTPUT_DIR = "data"
SEED       = 42    # Fixed seed for reproducibility — ensures train/val/test split is identical every run
VAL_SIZE   = 0.10  # 10% validation
TEST_SIZE  = 0.10  # 10% test (computed on the remaining data after first split)

SYSTEM_PROMPT = (
    "You are a medical assistant. Given a description of symptoms, "
    "identify the most likely disease. Answer with only the disease name."
)

def build_prompt(symptoms: str, label: str | None = None) -> str:
    """
    Format prompt using Mistral instruct template:
    <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>
    If label is None, returns inference prompt (no answer appended).
    """
    user_msg = f"Patient symptoms:\n{symptoms.strip()}"
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_msg} [/INST]"
    if label is not None:
        prompt += f" {label.strip()}</s>"
    return prompt


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Normalize column names to lowercase (handles 'Label'/'Text' variants)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "label" in df.columns and "text" in df.columns, (
        f"Expected columns 'label' and 'text'. Found: {list(df.columns)}"
    )

    # 2. Clean data
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip()
    label_corrections = {
    "allergy": "Allergy",
    "diabetes": "Diabetes",
    "drug reaction": "Drug Reaction",
    "urinary tract infection": "Urinary Tract Infection",
    "gastroesophageal reflux disease": "Gastroesophageal Reflux Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    }
    df["label"] = df["label"].replace(label_corrections)
    df["text"]  = df["text"].str.strip()
    df = df[df["text"].str.len() > 10]  # drop entries with very short descriptions

    print(f"[INFO] After cleaning: {len(df)} rows")
    print(f"[INFO] Unique diseases: {df['label'].nunique()}")
    print(df["label"].value_counts().to_string())

    # 3. Save label mapping (disease name → integer id)
    labels = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    print(f"\n[INFO] Label map saved ({len(labels)} classes)")

    # 4. Build prompts for each row
    df["prompt"] = df.apply(lambda r: build_prompt(r["text"], r["label"]), axis=1)

    # 5. Stratified train / val / test split (preserves class distribution in each subset)
    train_df, val_test_df = train_test_split(
        df, test_size=VAL_SIZE + TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)  # recompute test ratio within val+test
    val_df, test_df = train_test_split(
        val_test_df, test_size=relative_test, stratify=val_test_df["label"], random_state=SEED
    )

    print(f"\n[INFO] Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 6. Convert to HuggingFace DatasetDict and save to disk
    def to_hf(frame):
        return Dataset.from_dict({
            "text":  frame["prompt"].tolist(),
            "label": frame["label"].tolist()
        })

    ds = DatasetDict({
        "train":      to_hf(train_df),
        "validation": to_hf(val_df),
        "test":       to_hf(test_df)
    })
    ds.save_to_disk(os.path.join(OUTPUT_DIR, "dataset"))
    print(f"[INFO] Dataset saved to {OUTPUT_DIR}/dataset/")

    # 7. Sanity check — print first training example
    print("\n── First training prompt ──")
    print(ds["train"][0]["text"][:500])


if __name__ == "__main__":
    main()