"""Train the landmark Transformer; save the best checkpoint by validation accuracy."""

import time

import torch
import torch.nn.functional as F
import yaml

from src.dataset import make_dataloaders
from src.model import LandmarkTransformer


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()                                            # dropout off
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():                                   # no gradients needed
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                              # (B, C)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def main() -> None:
    cfg = yaml.safe_load(open("configs/config.yaml"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    train_dl, val_dl = make_dataloaders(cfg, batch_size=32)
    model = LandmarkTransformer(n_classes=len(cfg["gestures"]),
                                seq_len=cfg["seq_len"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs, best_acc = 40, 0.0
    for epoch in range(1, epochs + 1):
        model.train()                                       # dropout on
        t0, running = time.time(), 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)                              # forward pass
            loss = F.cross_entropy(logits, yb)              # scalar

            optimizer.zero_grad()                           # clear old gradients
            loss.backward()                                 # chain rule -> dL/dtheta for all theta
            optimizer.step()                                # theta <- theta - lr * (adapted) grad

            running += loss.item() * len(yb)

        train_loss = running / len(train_dl.dataset)
        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  ({time.time()-t0:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/transformer_best.pt")

    print(f"best val_acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()