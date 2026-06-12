# YieldFusionAI

[![DOI](https://zenodo.org/badge/1267307148.svg)](https://doi.org/10.5281/zenodo.20625406)

## YieldFusionAI: An Explainable Multi-Season Spatio-Temporal Deep Learning Framework for Accurate Crop Yield Forecasting

YieldFusionAI is an end-to-end deep learning framework for crop yield prediction using multi-season Sentinel-2 satellite imagery. The framework integrates spectral feature engineering, spatial feature extraction, temporal dependency modeling, attention-based learning, explainable artificial intelligence (XAI), and post-inference calibration to generate accurate, interpretable, and region-adaptive crop yield forecasts.

The framework is designed to support precision agriculture, agricultural monitoring, food security planning, and data-driven decision-making by leveraging remote sensing and deep learning technologies.

---

# Features

## Multi-Season Crop Yield Forecasting

- Utilizes Sentinel-2 multi-temporal imagery.
- Learns seasonal crop growth dynamics.
- Supports multi-year and multi-crop forecasting.

## Spectral Feature Engineering

- NDVI
- EVI
- SAVI
- GNDVI
- NDWI

## Spatial Feature Extraction

- Sobel edge enhancement
- Laplacian edge enhancement
- Texture-aware representation learning

## Deep Learning-Based Yield Prediction

- CNN spatial encoder
- Bi-LSTM temporal encoder
- Phenology-aware attention mechanism
- Fully connected regression head

## Explainable AI

- SHAP feature attribution
- Temporal attention visualization
- Yield-driving factor analysis

## Post-Inference Calibration

- Regional calibration
- Residual error correction
- Improved deployment robustness

## Evaluation Framework

- Five-fold cross-validation
- Temporal hold-out validation
- Region-wise validation
- Baseline comparisons

---

# Framework Architecture

```text
Sentinel-2 Imagery
        │
        ▼
Data Preprocessing
        │
        ▼
Vegetation Indices
(NDVI, EVI, SAVI, GNDVI, NDWI)
        │
        ▼
Spatial Feature Extraction
(Sobel + Laplacian)
        │
        ▼
PCA Feature Reduction
        │
        ▼
Temporal Tensor Construction
[T × 32 × 32 × C]
        │
        ▼
YieldNet-ST
(CNN + BiLSTM + Attention)
        │
        ▼
Yield Prediction
        │
        ▼
Calibration Module
        │
        ▼
Explainability Module
(SHAP + Attention)
        │
        ▼
Final Yield Forecast
```

# Directory Structure

```text
YieldFusionAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── shapefiles/
│   └── yield_labels.csv
│
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
│
├── notebooks/
│   └── YieldFusionAI_Colab.ipynb
│
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── maps/
│
├── requirements.txt
└── README.md
```

# Dataset Description

## Satellite Data

### Source

- Sentinel-2 Multispectral Instrument (MSI)

### Platform

- Copernicus Data Space Ecosystem
- Google Earth Engine

### Spatial Resolution

- 10 m
- 20 m

### Temporal Resolution

- 5-day revisit

### Bands Used

- B2 - Blue
- B3 - Green
- B4 - Red
- B5 - Red Edge
- B6 - Red Edge
- B7 - Red Edge
- B8 - NIR
- B8A - Narrow NIR
- B11 - SWIR
- B12 - SWIR

## Yield Data

Ground-truth crop yield information may be obtained from:

- Government agricultural statistics
- Crop production reports
- Agricultural survey datasets
- Regional agricultural databases

### Supported Crops

- Rice
- Maize
- Groundnut

The framework can be extended to:

- Wheat
- Cotton
- Soybean
- Sugarcane
- Millets
- Other crops

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YieldFusionAI.git
cd YieldFusionAI
```

## Create Environment

```bash
conda create -n yieldfusionai python=3.10
conda activate yieldfusionai
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Required Libraries

- tensorflow
- keras
- numpy
- pandas
- scikit-learn
- opencv-python
- matplotlib
- seaborn
- rasterio
- gdal
- geopandas
- shap
- scipy
- xgboost
- joblib
- tqdm

---

# Data Preparation

## Expected Input Tensor

```python
X.shape = (
    N,
    T,
    32,
    32,
    C
)
```

Where:

- N = number of samples
- T = seasonal time steps
- 32 × 32 = patch size
- C = spectral channels

Example:

```python
(432, 12, 32, 32, 15)
```

---

# Running the Framework

## Step 1: Build Dataset

```bash
python src/dataset_builder.py
```

## Step 2: Train YieldNet-ST

```bash
python src/train.py
```

Output:

```text
outputs/models/
```

## Step 3: Evaluate Model

```bash
python src/evaluate.py
```

Outputs:

- RMSE
- MAE
- MAPE
- R²

## Step 4: Run Explainability

```bash
python src/explainability.py
```

Outputs:

- SHAP summary plots
- Feature importance plots
- Temporal attention plots

## Step 5: Generate Visualizations

```bash
python src/visualization.py
```

Outputs:

- Prediction plots
- Training curves
- Calibration plots
- Yield maps

---

# YieldNet-ST Architecture

```text
Input Tensor
      │
      ▼
TimeDistributed Conv2D
      │
      ▼
ReLU
      │
      ▼
MaxPooling
      │
      ▼
Dropout
      │
      ▼
Flatten
      │
      ▼
Bi-LSTM
      │
      ▼
Attention Layer
      │
      ▼
Context Vector
      │
      ▼
Dense Layer
      │
      ▼
Dropout
      │
      ▼
Linear Regression Output
      │
      ▼
Predicted Yield
```

# Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)

# Baseline Models

- Random Forest
- Support Vector Regression
- XGBoost
- LSTM
- CNN-LSTM

Comparison is performed against:

- YieldNet-ST (Proposed)

---

# Explainable AI

## SHAP Analysis

Provides:

- Feature importance ranking
- Local prediction explanation
- Global model interpretation

## Temporal Attention Analysis

Provides:

- Stage-wise crop importance
- Phenological relevance estimation
- Growth-stage contribution analysis

---

# Calibration Module

Post-inference calibration includes:

- Regional Calibration
- Error-Aware Correction

Benefits:

- Reduced systematic bias
- Improved regional adaptation
- Better deployment performance

---

# Experimental Configuration

```text
Optimizer       : Adam
Learning Rate   : 0.001
Batch Size      : 32
Epochs          : 200
Dropout         : 0.3
Early Stopping  : 10 epochs
Validation      : 5-Fold Cross Validation
Loss Function   : MSE
```

---

# Output Artifacts

The framework automatically generates:

- Trained Model (.h5)
- Training Curves
- Prediction Results
- Cross-Validation Reports
- SHAP Explanations
- Attention Visualizations
- Calibration Reports
- Yield Prediction Maps

Stored in:

```text
outputs/
```

---

# Applications

- Precision Agriculture
- Crop Monitoring
- Agricultural Insurance
- Yield Estimation
- Food Security Planning
- Smart Farming
- Government Agricultural Planning
- Climate Impact Assessment
- Crop Advisory Systems

---

# Future Enhancements

- Transformer-based temporal modeling
- Multi-modal weather integration
- Soil and climate data fusion
- Federated agricultural learning
- Edge deployment
- Real-time yield forecasting
- UAV and hyperspectral integration
- Digital agriculture dashboards

---

# Citation

```bibtex
@software{YieldFusionAI2026,
  title={YieldFusionAI: An Explainable Multi-Season Spatio-Temporal Deep Learning Framework for Accurate Crop Yield Forecasting},
  author={Mahendra Somu},
  year={2026},
  doi={10.5281/zenodo.20625406}
}
```

---

# License

MIT License

Copyright (c) 2026 Mahendra Somu

This project is licensed under the MIT License. See the LICENSE file for details.

---

# Contact

For questions, collaborations, or research inquiries:

Author: Mahendra Somu

DOI:
https://doi.org/10.5281/zenodo.20625406

GitHub:
https://github.com/YOUR_USERNAME/YieldFusionAI

---

# Acknowledgement

This work leverages open-access Sentinel-2 satellite imagery provided by the European Space Agency (ESA) through the Copernicus Programme.

The authors acknowledge the contributions of the open-source scientific computing and geospatial communities whose tools enabled the development of YieldFusionAI.
