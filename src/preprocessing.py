import numpy as np

def cloud_mask_placeholder(image, cloud_probability=None, threshold=0.4):
    """Masks cloudy pixels. Replace with Sentinel-2 QA60/S2-cloud-probability logic for real data."""
    if cloud_probability is None:
        return image
    mask = cloud_probability < threshold
    return image * mask[..., None]

def roi_clip_placeholder(image, geometry=None):
    """Placeholder for rasterio/geopandas clipping; returns image unchanged in demo mode."""
    return image

def temporal_align_placeholder(images, expected_steps):
    if len(images) == expected_steps:
        return np.stack(images)
    if len(images) > expected_steps:
        return np.stack(images[:expected_steps])
    pad = [images[-1]] * (expected_steps - len(images))
    return np.stack(images + pad)
