import numpy as np
import matplotlib.pyplot as plt
from .model_yieldnet_st import build_attention_model

def attention_scores(model, X):
    extractor = build_attention_model(model)
    return extractor.predict(X, verbose=0)

def plot_attention(weights, out_path):
    avg = np.mean(weights, axis=0)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(avg)+1), avg, marker="o")
    plt.xlabel("Time step")
    plt.ylabel("Normalized attention")
    plt.title("Phenology-Aware Temporal Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def permutation_feature_importance(model, X, y, scaler=None, channel_names=None, n_repeats=2):
    base = model.predict(X, verbose=0)
    if scaler is not None:
        base = scaler.inverse_transform(base)
        yy = scaler.inverse_transform(y)
    else:
        yy = y
    base_mae = np.mean(np.abs(yy.reshape(-1) - base.reshape(-1)))
    rng = np.random.default_rng(42)
    scores = []
    for c in range(X.shape[-1]):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            flat = Xp[..., c].reshape(Xp.shape[0], -1)
            rng.shuffle(flat, axis=0)
            Xp[..., c] = flat.reshape(Xp[..., c].shape)
            pred = model.predict(Xp, verbose=0)
            if scaler is not None:
                pred = scaler.inverse_transform(pred)
            mae = np.mean(np.abs(yy.reshape(-1) - pred.reshape(-1)))
            deltas.append(mae - base_mae)
        scores.append(np.mean(deltas))
    names = channel_names or [f"C{i}" for i in range(X.shape[-1])]
    return names, np.asarray(scores)

def plot_feature_importance(names, scores, out_path, top_k=15):
    idx = np.argsort(scores)[::-1][:top_k]
    plt.figure(figsize=(8, 5))
    plt.bar([names[i] for i in idx], scores[idx])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAE increase after permutation")
    plt.title("Feature Attribution Approximation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
