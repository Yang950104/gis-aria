# =============================================================================
# Week 12: Setup & Task 1 - K-means Unsupervised Classification
# Study Area: Xiulin / Taroko Region
# ARIA v8.0 Classification Engine
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pystac_client
import planetary_computer
import stackstac
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Setup & Data Fetching
# =============================================================================

# Define Taroko bounding box
TAROKO_BBOX = [121.40, 24.10, 121.80, 24.25]

print("=" * 60)
print("Setup & Data Fetching - Xiulin/Taroko Region")
print("=" * 60)
print(f"Bounding Box: {TAROKO_BBOX}")
print()

# Initialize STAC catalog
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Progressive search configurations
search_configs = [
    ("2024-04-15/2024-05-31", 20, "Phase 1: 4/15-5/31, cc<20%"),
    ("2024-04-03/2024-08-31", 30, "Phase 2: 4/3-8/31, cc<30%"),
    ("2024-04-03/2024-12-31", 50, "Phase 3: 4/3-12/31, cc<50%"),
]

items = None
for dt_range, max_cc, desc in search_configs:
    print(f"Trying {desc}...")
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=TAROKO_BBOX,
            datetime=dt_range,
            query={"eo:cloud_cover": {"lt": max_cc}},
        )
        items = search.item_collection()
        print(f"  → Found {len(items)} scenes")
        if len(items) > 0:
            break
    except Exception as e:
        print(f"  → Error: {e}")
        continue

if items is None or len(items) == 0:
    raise RuntimeError("No Sentinel-2 scenes found. Please check network or try different parameters.")

# Select scene with lowest cloud cover
items_sorted = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])
best_item = items_sorted[0]

print("\n=== Selected Scene ===")
print(f"  Scene ID    : {best_item.id}")
print(f"  Date        : {best_item.datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Cloud Cover : {best_item.properties['eo:cloud_cover']:.1f}%")
print(f"  Platform    : {best_item.properties.get('platform', 'N/A')}")

# Verify required bands
BANDS_ALL = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
available_bands = list(best_item.assets.keys())
missing_bands = [band for band in BANDS_ALL if band not in available_bands]

if missing_bands:
    print(f"\n⚠️ Missing bands: {missing_bands}")
else:
    print(f"✓ All required bands available")

print("\n" + "=" * 60)
print("Loading data with stackstac...")
print("=" * 60)

# Sign the item and load data
signed_item = planetary_computer.sign(best_item)

# Stream data using stackstac
stack = stackstac.stack(
    [signed_item],
    assets=BANDS_ALL,
    epsg=32651,  # UTM 51N for Taiwan east coast
    resolution=20,
    bounds_latlon=TAROKO_BBOX,
    dtype="float64",
)

print(f"Lazy array shape: {stack.shape} (time, band, y, x)")
print("Computing data...")

# Compute the data
data = stack.compute()
img_all = data.values[0]  # shape: (7, H, W)

print(f"Data loaded successfully!")
print(f"Shape: {img_all.shape}")

# =============================================================================
# 2. Preprocessing (Crucial)
# =============================================================================

print("\n" + "=" * 60)
print("Preprocessing - Cloud Mask & Reflectance Conversion")
print("=" * 60)

# Separate SCL and reflectance bands
scl = img_all[-1]  # Last band = SCL
img = img_all[:-1] / 10000.0  # Convert to surface reflectance

# Apply Cloud Mask using SCL band
# SCL values: 3=Cloud Shadow, 8=Cloud (medium), 9=Cloud (high), 10=Cirrus, 11=Snow
cloud_mask = np.isin(scl, [3, 8, 9, 10, 11])
n_cloud = int(np.sum(cloud_mask))
n_total = cloud_mask.size

print(f"\n=== SCL Cloud Mask ===")
print(f"  Cloud/Shadow/Snow pixels: {n_cloud:,} / {n_total:,} ({n_cloud/n_total*100:.1f}%)")

# Set cloud/shadow/snow pixels to NaN
img[:, cloud_mask] = np.nan

# Also set invalid values (<= 0 or > 1) to NaN
img = np.where((img <= 0) | (img > 1), np.nan, img)

n_bands, n_rows, n_cols = img.shape

print(f"\nImage shape: {img.shape} (bands, height, width)")
print(f"Pixel size: 20m x 20m")
print(f"Image dimensions: {n_cols * 20 / 1000:.1f} km x {n_rows * 20 / 1000:.1f} km")
print(f"Valid pixel ratio (after cloud mask): {np.sum(~np.isnan(img[0])) / img[0].size * 100:.1f}%")

# Print band statistics
print("\n=== Band Statistics ===")
for i, name in enumerate(BAND_NAMES):
    valid = img[i][~np.isnan(img[i])]
    if len(valid) > 0:
        print(
            f"  {name:6s} ({BANDS[i]}): "
            f"min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}"
        )

# Reshape image into 2D feature matrix X of shape (n_pixels, 6)
X = img.reshape(n_bands, -1).T  # shape: (n_pixels, 6)

# Filter out rows containing NaNs to create X_valid
valid_pixel_mask = ~np.any(np.isnan(X), axis=1)
X_valid = X[valid_pixel_mask]

print(f"\n=== Feature Matrix ===")
print(f"Total pixels: {X.shape[0]:,}")
print(f"Valid pixels: {X_valid.shape[0]:,}")
print(f"Feature dimensions: {X_valid.shape[1]} bands")

# =============================================================================
# 3. K-means Clustering (Task 1)
# =============================================================================

print("\n" + "=" * 60)
print("Task 1: K-means Unsupervised Classification")
print("=" * 60)

# Run K-means with specified parameters
K = 5
print(f"\nRunning K-means (K={K})...")
print(f"Parameters: n_clusters={K}, random_state=42, n_init=10")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_valid)

print(f"K-means completed! Inertia = {kmeans.inertia_:.2f}")

# Reconstruct the 2D label map (restoring NaN masked areas)
class_map = np.full(X.shape[0], np.nan)
class_map[valid_pixel_mask] = labels
class_map = class_map.reshape(n_rows, n_cols)

# Plot the classification result
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(class_map, cmap='tab10', vmin=-0.5, vmax=K-0.5)
cbar = plt.colorbar(im, ax=ax, ticks=range(K), shrink=0.8)
cbar.set_label("Cluster ID", fontsize=12)
ax.set_title(f"K-means Unsupervised Classification (K={K})\nXiulin/Taroko Region", fontsize=14)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "kmeans_classification.png"), dpi=300, bbox_inches="tight")
plt.show()

# Print cluster statistics
print("\n=== Cluster Statistics ===")
for c in range(K):
    count = np.sum(labels == c)
    print(f"  Cluster {c}: {count:>8,} pixels ({count / len(labels) * 100:.1f}%)")

# =============================================================================
# 4. Calculate and Print Mean Reflectance Values for Each Cluster
# =============================================================================

print("\n" + "=" * 60)
print("Mean Reflectance Values for Each Cluster")
print("=" * 60)

# Calculate mean reflectance for each cluster
cluster_means = np.zeros((K, n_bands))
for c in range(K):
    cluster_pixels = X_valid[labels == c]
    cluster_means[c] = cluster_pixels.mean(axis=0)

# Print mean reflectance table
print(f"\n{'Cluster':<10} ", end="")
for i, name in enumerate(BAND_NAMES):
    print(f"{name:>10s} ", end="")
print()

print("-" * (10 + 11 * n_bands))
for c in range(K):
    print(f"{c:<10} ", end="")
    for i in range(n_bands):
        print(f"{cluster_means[c, i]:>10.4f} ", end="")
    print()

# Plot spectral signatures
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, K))

for c in range(K):
    ax.plot(
        range(n_bands),
        cluster_means[c],
        'o-',
        color=colors[c],
        linewidth=2,
        markersize=8,
        label=f"Cluster {c}"
    )

ax.set_xticks(range(n_bands))
ax.set_xticklabels([f"{BAND_NAMES[i]}\n({BANDS[i]})" for i in range(n_bands)], fontsize=11)
ax.set_ylabel("Mean Reflectance", fontsize=13)
ax.set_xlabel("Band", fontsize=13)
ax.set_title("Cluster Mean Spectral Signatures\nXiulin/Taroko Region", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "kmeans_spectral_signatures.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("Task 1 Completed Successfully!")
print("=" * 60)
print("\nSpectral Interpretation Guide:")
print("  - High NIR, Low Red → Healthy Vegetation")
print("  - Low in all bands → Water")
print("  - High Red/SWIR → Bare Soil / Landslide")
print("  - Overall bright → Built-up / Artificial surfaces")
print("\nUse the spectral signatures above to manually label each cluster.")
