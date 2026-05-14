import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "models/mistral-7b-instruct-v0.3"
LORA_WEIGHTS = "outputs/mistral-disease-lora/final"
MAX_NEW_TOKENS = 20
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a medical assistant. Given a description of symptoms, "
    "identify the most likely disease. Answer with only the disease name."
)


class DiseasePredictor:
    def __init__(self):
        print("[INFO] Loading model ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
        self.model.eval()
        print("Model ready.\n")

    def build_prompt(self, symptoms: str) -> str:
        user_msg = f"Patient symptoms:\n{symptoms.strip()}"
        return f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_msg} [/INST]"

    def predict(self, symptoms: str) -> str:
        prompt = self.build_prompt(symptoms)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)
        response = decoded.split("[/INST]")[-1]
        response = re.sub(r"</s>.*", "", response, flags=re.DOTALL)
        return response.split("\n")[0].strip()


def main():
    predictor = DiseasePredictor()

    print("─" * 55)
    print("  Disease Predictor — Fine-tuned Mistral-7B")
    print("  Type 'quit' to exit.")
    print("─" * 55)

    while True:
        print()
        symptoms = input("Describe symptoms: ").strip()
        if symptoms.lower() in {"quit", "exit", "q"}:
            break
        if not symptoms:
            continue
        prediction = predictor.predict(symptoms)
        print(f"\n  → Predicted disease: {prediction}\n")


if __name__ == "__main__":
    main()