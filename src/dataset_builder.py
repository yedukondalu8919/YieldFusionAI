import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from .config import CFG
from .feature_engineering import construct_feature_tensor

CROPS = ["rice", "maize", "groundnut"]
SEASONS = ["kharif", "rabi"]
REGIONS = [f"R{i:02d}" for i in range(1, 19)]

def generate_synthetic_raw_dataset(cfg=CFG):
    rng = np.random.default_rng(cfg.random_seed)
    X, y, meta = [], [], []
    for i in range(cfg.n_samples):
        crop_id = i % len(CROPS)
        season_id = (i // len(CROPS)) % len(SEASONS)
        region = REGIONS[i % len(REGIONS)]
        year = 2016 + (i % 7)
        base = rng.uniform(0.12, 0.35, size=(cfg.patch_size, cfg.patch_size, cfg.raw_bands))
        seq = []
        seasonal_peak = rng.uniform(0.45, 0.95)
        for t in range(cfg.time_steps):
            phase = np.sin(np.pi * (t + 1) / cfg.time_steps)
            img = base + phase * seasonal_peak * rng.uniform(0.05, 0.18, size=base.shape)
            img[..., 6] += phase * rng.uniform(0.25, 0.45)  # NIR growth
            img[..., 2] -= phase * rng.uniform(0.02, 0.05)  # Red absorption
            img += rng.normal(0, 0.015, img.shape)
            seq.append(np.clip(img, 0, 1).astype(np.float32))
        raw = np.stack(seq, axis=0)
        ft, _ = construct_feature_tensor(raw, cfg.pca_components)
        crop_effect = [0.8, 0.55, 0.45][crop_id]
        season_effect = [0.22, -0.05][season_id]
        nd_signal = ft[..., 0].mean()
        texture_signal = ft[..., cfg.pca_components:].mean()
        yield_val = 2.2 + 3.0 * nd_signal + 0.7 * texture_signal + crop_effect + season_effect + rng.normal(0, 0.15)
        X.append(ft)
        y.append(yield_val)
        meta.append({"sample_id": i, "crop": CROPS[crop_id], "season": SEASONS[season_id], "region": region, "year": year})
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y, pd.DataFrame(meta)

def save_processed_dataset(cfg=CFG):
    cfg.make_dirs()
    X, y, meta = generate_synthetic_raw_dataset(cfg)
    np.save(cfg.processed_dir / "X.npy", X)
    np.save(cfg.processed_dir / "y.npy", y)
    meta.to_csv(cfg.processed_dir / "metadata.csv", index=False)
    return X, y, meta

def load_or_create_dataset(cfg=CFG):
    x_path, y_path, m_path = cfg.processed_dir / "X.npy", cfg.processed_dir / "y.npy", cfg.processed_dir / "metadata.csv"
    if x_path.exists() and y_path.exists() and m_path.exists():
        return np.load(x_path), np.load(y_path), pd.read_csv(m_path)
    return save_processed_dataset(cfg)

def scale_targets(y_train, y_val=None, y_test=None):
    scaler = MinMaxScaler()
    y_train_s = scaler.fit_transform(y_train)
    outs = [y_train_s]
    for y in [y_val, y_test]:
        if y is not None:
            outs.append(scaler.transform(y))
    return (*outs, scaler)

def crop_season_strata(meta):
    return (meta["crop"].astype(str) + "_" + meta["season"].astype(str)).values

def make_kfold_indices(meta, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = crop_season_strata(meta)
    return list(skf.split(np.zeros(len(meta)), strata))

def make_temporal_holdout(meta, train_year_max=2020):
    train_idx = np.where(meta["year"].values <= train_year_max)[0]
    test_idx = np.where(meta["year"].values > train_year_max)[0]
    return train_idx, test_idx

def make_region_holdout(meta, holdout_regions=None):
    if holdout_regions is None:
        holdout_regions = sorted(meta["region"].unique())[-3:]
    test_mask = meta["region"].isin(holdout_regions).values
    return np.where(~test_mask)[0], np.where(test_mask)[0]
