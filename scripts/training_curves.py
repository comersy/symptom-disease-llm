import json
import matplotlib.pyplot as plt

TRAINER_STATE = "outputs/mistral-disease-lora/checkpoint-600/trainer_state.json"

with open(TRAINER_STATE) as f:
    state = json.load(f)

logs = state["log_history"]

train_steps, train_loss = [], []
eval_steps, eval_loss = [], []

for entry in logs:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(train_steps, train_loss, label="Train loss", color="steelblue")
axes[0].plot(eval_steps, eval_loss, label="Eval loss", color="tomato", marker="o")
axes[0].set_title("Loss")
axes[0].set_xlabel("Steps")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Token accuracy
train_acc_steps, train_acc = [], []
eval_acc_steps, eval_acc = [], []

for entry in logs:
    if "mean_token_accuracy" in entry and "loss" in entry:
        train_acc_steps.append(entry["step"])
        train_acc.append(entry["mean_token_accuracy"])
    if "eval_mean_token_accuracy" in entry:
        eval_acc_steps.append(entry["step"])
        eval_acc.append(entry["eval_mean_token_accuracy"])

axes[1].plot(train_acc_steps, train_acc, label="Train accuracy", color="steelblue")
axes[1].plot(eval_acc_steps, eval_acc, label="Eval accuracy", color="tomato", marker="o")
axes[1].set_title("Token Accuracy")
axes[1].set_xlabel("Steps")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150)
plt.show()
print("[✓] Saved to outputs/training_curves.png")