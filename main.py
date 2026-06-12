from src.config import CFG
from src.dataset_builder import save_processed_dataset
from src.train import train_proposed_single_split, train_baselines

if __name__ == "__main__":
    CFG.make_dirs()
    print("Creating/loading data...")
    save_processed_dataset(CFG)
    print("Training proposed YieldNet-ST...")
    _, raw, cal = train_proposed_single_split(CFG)
    print("Raw metrics:", raw)
    print("Calibrated metrics:", cal)
    print("Training baselines...")
    print(train_baselines(CFG))
    print("Done. See outputs/ for models, metrics, and plots.")
