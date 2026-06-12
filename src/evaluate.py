import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from .config import CFG

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def plot_actual_vs_pred(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.75)
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Actual vs Predicted Crop Yield")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_loss(history, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history.get("loss", []), label="Training loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_metric_bars(df, out_path, metric="RMSE"):
    plt.figure(figsize=(8, 4))
    plt.bar(df["Model"], df[metric])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"Baseline Comparison by {metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
