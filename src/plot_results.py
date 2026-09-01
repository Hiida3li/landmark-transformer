"""Plot validation accuracy curves from saved training histories."""

import json

import matplotlib.pyplot as plt

RUNS = {"transformer": "Transformer (73K params)", "mlp": "MLP (25K params)"}

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
for name, label in RUNS.items():
    hist = json.load(open(f"results/{name}_history.json"))
    ax.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist],
            label=label, linewidth=1.8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy")
ax.set_title("Validation accuracy: Transformer vs MLP baseline")
ax.set_ylim(0.5, 1.02)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/val_accuracy.png")
print("saved results/val_accuracy.png")