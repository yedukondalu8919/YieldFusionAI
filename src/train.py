import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from joblib import dump
from .config import CFG
from .dataset_builder import load_or_create_dataset, make_kfold_indices, scale_targets
from .model_yieldnet_st import build_yieldnet_st
from .evaluate import regression_metrics, save_metrics, plot_actual_vs_pred, plot_loss, plot_metric_bars
from .calibration import RegionalCalibrator
from .explainability import attention_scores, plot_attention, permutation_feature_importance, plot_feature_importance

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_proposed_single_split(cfg=CFG):
    set_seed(cfg.random_seed)
    cfg.make_dirs()
    X, y, meta = load_or_create_dataset(cfg)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg.random_seed,
                                           stratify=meta["crop"].astype(str)+"_"+meta["season"].astype(str))
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2, random_state=cfg.random_seed,
                                          stratify=(meta.iloc[idx_train]["crop"].astype(str)+"_"+meta.iloc[idx_train]["season"].astype(str)))
    y_train_s, y_val_s, y_test_s, scaler = scale_targets(y[idx_train], y[idx_val], y[idx_test])
    model = build_yieldnet_st(X.shape[1:], cfg.dropout, cfg.l2_weight, cfg.learning_rate)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5)]
    history = model.fit(X[idx_train], y_train_s, validation_data=(X[idx_val], y_val_s), epochs=cfg.epochs,
                        batch_size=cfg.batch_size, callbacks=callbacks, verbose=2)
    val_raw = scaler.inverse_transform(model.predict(X[idx_val], verbose=0))
    test_raw = scaler.inverse_transform(model.predict(X[idx_test], verbose=0))
    cal = RegionalCalibrator().fit(val_raw, y[idx_val], meta.iloc[idx_val]["region"].values)
    test_cal = cal.predict(test_raw, meta.iloc[idx_test]["region"].values)
    metrics_raw = regression_metrics(y[idx_test], test_raw)
    metrics_cal = regression_metrics(y[idx_test], test_cal)
    model.save(cfg.models_dir / "yieldnet_st.keras")
    dump(scaler, cfg.models_dir / "target_scaler.joblib")
    dump(cal, cfg.models_dir / "regional_calibrator.joblib")
    save_metrics({"raw": metrics_raw, "calibrated": metrics_cal}, cfg.metrics_dir / "proposed_single_split_metrics.json")
    plot_loss(history, cfg.plots_dir / "training_loss.png")
    plot_actual_vs_pred(y[idx_test].reshape(-1), test_cal.reshape(-1), cfg.plots_dir / "actual_vs_predicted_calibrated.png")
    weights = attention_scores(model, X[idx_test][: min(64, len(idx_test))])
    plot_attention(weights, cfg.plots_dir / "attention_temporal_importance.png")
    names, scores = permutation_feature_importance(model, X[idx_test][: min(40, len(idx_test))], y_test_s[: min(40, len(idx_test))], scaler=scaler)
    plot_feature_importance(names, scores, cfg.plots_dir / "feature_importance_permutation.png")
    return model, metrics_raw, metrics_cal

def flatten_for_ml(X):
    return X.reshape(X.shape[0], -1)

def train_baselines(cfg=CFG):
    cfg.make_dirs()
    X, y, meta = load_or_create_dataset(cfg)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg.random_seed,
                                           stratify=meta["crop"].astype(str)+"_"+meta["season"].astype(str))
    Xtr, Xte = flatten_for_ml(X[idx_train]), flatten_for_ml(X[idx_test])
    ytr, yte = y[idx_train].reshape(-1), y[idx_test].reshape(-1)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=cfg.random_seed, n_jobs=-1),
        "SVR": SVR(C=10, epsilon=0.05, kernel="rbf"),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=cfg.random_seed, objective="reg:squarederror")
    rows = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        met = regression_metrics(yte, pred)
        rows.append({"Model": name, **met})
        dump(m, cfg.models_dir / f"{name.lower()}_baseline.joblib")
    df = pd.DataFrame(rows)
    df.to_csv(cfg.metrics_dir / "baseline_metrics.csv", index=False)
    if not df.empty:
        plot_metric_bars(df, cfg.plots_dir / "baseline_rmse_comparison.png", "RMSE")
    return df

def run_cross_validation(cfg=CFG):
    set_seed(cfg.random_seed)
    X, y, meta = load_or_create_dataset(cfg)
    rows = []
    for fold, (tr, te) in enumerate(make_kfold_indices(meta, cfg.folds, cfg.random_seed), start=1):
        tr, val = train_test_split(tr, test_size=0.2, random_state=cfg.random_seed,
                                   stratify=(meta.iloc[tr]["crop"].astype(str)+"_"+meta.iloc[tr]["season"].astype(str)))
        ytr_s, yval_s, yte_s, scaler = scale_targets(y[tr], y[val], y[te])
        model = build_yieldnet_st(X.shape[1:], cfg.dropout, cfg.l2_weight, cfg.learning_rate)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)]
        model.fit(X[tr], ytr_s, validation_data=(X[val], yval_s), epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=cb, verbose=0)
        pred = scaler.inverse_transform(model.predict(X[te], verbose=0))
        met = regression_metrics(y[te], pred)
        rows.append({"Fold": fold, **met})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.metrics_dir / "cross_validation_metrics.csv", index=False)
    save_metrics({"mean": df.drop(columns=["Fold"]).mean().to_dict(), "std": df.drop(columns=["Fold"]).std().to_dict()}, cfg.metrics_dir / "cross_validation_summary.json")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YieldFusionAI modules")
    parser.add_argument("--mode", choices=["proposed", "baselines", "cv", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for quick tests")
    args = parser.parse_args()
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.mode in ["proposed", "all"]:
        print(train_proposed_single_split(CFG)[1:])
    if args.mode in ["baselines", "all"]:
        print(train_baselines(CFG))
    if args.mode in ["cv", "all"]:
        print(run_cross_validation(CFG))
