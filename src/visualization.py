import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_geospatial_yield_placeholder(meta, predictions, out_path):
    df = meta.copy()
    df["prediction"] = np.asarray(predictions).reshape(-1)
    pivot = df.groupby("region")["prediction"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.bar(pivot["region"], pivot["prediction"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean predicted yield")
    plt.title("Region-Level Yield Map Placeholder")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return pivot
