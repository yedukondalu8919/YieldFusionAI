# YieldFusionAI

YieldFusionAI is a complete Python/TensorFlow implementation of an explainable multi-season spatio-temporal crop yield forecasting framework.

## Implemented Components

- Synthetic Sentinel-2-like demo dataset generator
- Vegetation index computation: NDVI, EVI, SAVI, GNDVI, NDWI
- PCA-based channel reduction
- Sobel and Laplacian edge feature extraction
- Multi-season tensor construction `[samples, T, 32, 32, C]`
- YieldNet-ST model: TimeDistributed CNN + Bi-LSTM + phenology-aware attention
- Post-inference regional calibration
- Baseline models: Random Forest, SVR, XGBoost if installed
- Evaluation metrics: RMSE, MAE, R², MAPE
- Explainability plots: temporal attention and permutation-based feature attribution
- Cross-validation support

## Project Structure

```text
YieldFusionAI/
├── data/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── dataset_builder.py
│   ├── model_yieldnet_st.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibration.py
│   ├── explainability.py
│   └── visualization.py
├── outputs/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Quick Run

For a fast test:

```bash
python -m src.train --mode proposed --epochs 3
```

Run the full demo pipeline:

```bash
python main.py
```

Run baselines:

```bash
python -m src.train --mode baselines
```

Run five-fold cross-validation:

```bash
python -m src.train --mode cv --epochs 20
```

## Real Sentinel-2 Data Integration

The current package is fully functional with synthetic Sentinel-2-like tensors. To use real data, replace the synthetic generator in `src/dataset_builder.py` with raster loading logic. The expected input per sample is:

```text
T × 32 × 32 × 10
```

where bands follow:

```text
B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
```

The feature construction module will automatically generate indices, PCA channels, edge features, and normalized tensors.

## Outputs

After execution, generated results are saved in:

```text
outputs/models/
outputs/metrics/
outputs/plots/
```

Main plots include:

- `training_loss.png`
- `actual_vs_predicted_calibrated.png`
- `attention_temporal_importance.png`
- `feature_importance_permutation.png`
- `baseline_rmse_comparison.png`

## Notes

For SCI/Scopus-style experimentation, use real Sentinel-2 Level-2A data, official yield records, region-wise holdout, temporal holdout, and five-fold cross-validation. The implementation is modular, so real geospatial preprocessing can be plugged into `src/preprocessing.py` and `src/dataset_builder.py`.
