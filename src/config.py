from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    processed_dir: Path = data_dir / "processed"
    outputs_dir: Path = project_root / "outputs"
    models_dir: Path = outputs_dir / "models"
    plots_dir: Path = outputs_dir / "plots"
    metrics_dir: Path = outputs_dir / "metrics"
    maps_dir: Path = outputs_dir / "maps"

    n_samples: int = 432
    time_steps: int = 12
    patch_size: int = 32
    raw_bands: int = 10
    pca_components: int = 5
    final_channels: int = 15
    random_seed: int = 42

    batch_size: int = 32
    epochs: int = 200
    patience: int = 10
    learning_rate: float = 1e-3
    dropout: float = 0.30
    l2_weight: float = 1e-4
    folds: int = 5

    calibration_lambda: float = 0.15
    calibration_epsilon: float = 0.25

    def make_dirs(self):
        for p in [self.processed_dir, self.outputs_dir, self.models_dir, self.plots_dir, self.metrics_dir, self.maps_dir]:
            p.mkdir(parents=True, exist_ok=True)

CFG = Config()
