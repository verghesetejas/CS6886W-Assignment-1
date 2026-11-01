import argparse
import csv
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

# Ensure imports work both with `python -m training.train` and W&B's script launcher.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vgg6 import VGG6
from data.cifar10 import loaders
from utils.seed import seed_all
from training.optim import make_optimizer
from training.engine import epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--activation")
    p.add_argument("--optimizer")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    return p.parse_args()


def save_curves(history, output_dir):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots = [
        ("train_loss", "Training loss"),
        ("train_accuracy", "Training accuracy"),
        ("val_loss", "Validation loss"),
        ("val_accuracy", "Validation accuracy"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.plot(epochs, [r[key] for r in history], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "curves.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CLI values override YAML; W&B sweep values override both.
    for value, key in [
        (args.activation, "activation"),
        (args.optimizer, "optimizer"),
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.lr, "learning_rate"),
    ]:
        if value is not None:
            cfg[key] = value

    sweep_mode = bool(os.getenv("WANDB_SWEEP_ID"))
    wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", False)) or sweep_mode
    run = None
    if wandb_enabled:
        wb_cfg = cfg.get("wandb", {})
        run = wandb.init(
            project=wb_cfg.get("project", cfg.get("wandb_project", "cs6886w-vgg6")),
            entity=wb_cfg.get("entity"),
            config=cfg,
            save_code=True,
        )
        # Sweep agent configuration is authoritative.
        for key in ["activation", "optimizer", "batch_size", "learning_rate", "epochs", "seed"]:
            if key in wandb.config:
                cfg[key] = wandb.config[key]

    seed_all(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    train_loader, val_loader, _ = loaders(
        cfg["data_dir"], int(cfg["batch_size"]), int(cfg["num_workers"]),
        int(cfg["seed"]), int(cfg["cutout_size"])
    )

    model = VGG6(act=cfg["activation"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        cfg["optimizer"], model.parameters(), float(cfg["learning_rate"]),
        float(cfg["weight_decay"]), float(cfg["momentum"])
    )
    amp_enabled = bool(cfg["use_amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    history = []
    best_accuracy = -1.0

    for ep in range(1, int(cfg["epochs"]) + 1):
        train_loss, train_accuracy = epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, amp=amp_enabled
        )
        val_loss, val_accuracy = epoch(
            model, val_loader, criterion, device, amp=amp_enabled
        )

        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(row)

        if run:
            # These epoch-level metrics generate W&B history charts.
            wandb.log(row, step=ep)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = os.path.join(output_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": ep,
                    "best_val_accuracy": best_accuracy,
                },
                checkpoint_path,
            )

    csv_path = os.path.join(output_dir, "history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    curves_path = save_curves(history, output_dir)
    if run:
        wandb.summary["best_val_accuracy"] = best_accuracy
        wandb.log({"best_val_accuracy": best_accuracy})
        wandb.save(csv_path)
        wandb.save(curves_path)
        wandb.finish()

    print(f"Best validation accuracy: {best_accuracy:.3f}%")
    print(f"Saved checkpoint, history and curves under: {output_dir}")


if __name__ == "__main__":
    main()
