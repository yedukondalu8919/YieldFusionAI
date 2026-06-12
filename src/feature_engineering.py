import numpy as np
from sklearn.decomposition import PCA
try:
    import cv2
except Exception:
    cv2 = None

EPS = 1e-6

def ndvi(nir, red):
    return (nir - red) / (nir + red + EPS)

def evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + EPS)

def savi(nir, red, L=0.5):
    return ((nir - red) / (nir + red + L + EPS)) * (1.0 + L)

def gndvi(nir, green):
    return (nir - green) / (nir + green + EPS)

def ndwi(green, nir):
    return (green - nir) / (green + nir + EPS)

def compute_indices(bands):
    """bands: H x W x 10 ordered B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12."""
    blue, green, red = bands[..., 0], bands[..., 1], bands[..., 2]
    nir = bands[..., 6]
    out = [ndvi(nir, red), evi(nir, red, blue), savi(nir, red), gndvi(nir, green), ndwi(green, nir)]
    return np.stack(out, axis=-1).astype(np.float32)

def sobel_laplacian_features(feature_map):
    chans = []
    for c in range(feature_map.shape[-1]):
        img = feature_map[..., c].astype(np.float32)
        if cv2 is not None:
            sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            sob = np.sqrt(sx ** 2 + sy ** 2)
            lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        else:
            gy, gx = np.gradient(img)
            sob = np.sqrt(gx ** 2 + gy ** 2)
            lap = np.gradient(gx)[1] + np.gradient(gy)[0]
        chans.extend([sob, lap])
    return np.stack(chans, axis=-1).astype(np.float32)

def minmax_scale_array(x, eps=1e-6):
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)

def pca_reduce_time_patch(feature_tensor, n_components=5):
    """feature_tensor: T x H x W x C; PCA fitted across all pixels/time."""
    T, H, W, C = feature_tensor.shape
    flat = feature_tensor.reshape(-1, C)
    n_components = min(n_components, C)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(flat).reshape(T, H, W, n_components).astype(np.float32)
    return reduced, pca

def construct_feature_tensor(raw_tensor, pca_components=5):
    """raw_tensor: T x H x W x 10 Sentinel-like bands."""
    vi = np.stack([compute_indices(raw_tensor[t]) for t in range(raw_tensor.shape[0])], axis=0)
    reduced, pca = pca_reduce_time_patch(vi, pca_components)
    edge = np.stack([sobel_laplacian_features(reduced[t]) for t in range(reduced.shape[0])], axis=0)
    enhanced = np.concatenate([reduced, edge], axis=-1)
    enhanced = minmax_scale_array(enhanced)
    return enhanced.astype(np.float32), pca
