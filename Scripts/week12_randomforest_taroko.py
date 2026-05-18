# =============================================================================
# Week 12: Task 2 - Random Forest Supervised Classification
# Study Area: Xiulin / Taroko Region
# ARIA v8.0 Classification Engine
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import median_filter
import pystac_client
import planetary_computer
import stackstac
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. Load Image Data (from Task 1)
# =============================================================================

print("=" * 60)
print("Loading Sentinel-2 Image Data")
print("=" * 60)

# Define Taroko bounding box
TAROKO_BBOX = [121.40, 24.10, 121.80, 24.25]

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
    raise RuntimeError("No Sentinel-2 scenes found.")

# Select scene with lowest cloud cover
items_sorted = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])
best_item = items_sorted[0]

print(f"\nSelected Scene: {best_item.id}")
print(f"Date: {best_item.datetime.strftime('%Y-%m-%d')}")
print(f"Cloud Cover: {best_item.properties['eo:cloud_cover']:.1f}%")

# Bands
BANDS_ALL = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

# Load data
signed_item = planetary_computer.sign(best_item)
stack = stackstac.stack(
    [signed_item],
    assets=BANDS_ALL,
    epsg=32651,
    resolution=20,
    bounds_latlon=TAROKO_BBOX,
    dtype="float64",
)

print("Loading data...")
data = stack.compute()
img_all = data.values[0]

# Preprocessing
scl = img_all[-1]
img = img_all[:-1] / 10000.0

# Cloud mask
cloud_mask = np.isin(scl, [3, 8, 9, 10, 11])
img[:, cloud_mask] = np.nan
img = np.where((img <= 0) | (img > 1), np.nan, img)

n_bands, n_rows, n_cols = img.shape

# Get transform from stackstac data
x_coords = data.x.values
y_coords = data.y.values
x_res = float(x_coords[1] - x_coords[0])
y_res = float(y_coords[1] - y_coords[0])
img_transform = from_bounds(
    x_coords[0], y_coords[-1], x_coords[-1], y_coords[0],
    n_cols, n_rows
)
img_crs = 'EPSG:32651'

print(f"Image shape: {img.shape}")
print(f"Image CRS: {img_crs}")

# =============================================================================
# 1. Robust KML Parsing and Rasterization
# =============================================================================

print("\n" + "=" * 60)
print("Task 2: Random Forest Supervised Classification")
print("=" * 60)
print("\nStep 1: Parse KML and Rasterize Training Samples")
print("-" * 60)

# KML file path
kml_file = "training_samples_taroco.kml"

# Parse KML using ElementTree
tree = ET.parse(kml_file)
root = tree.getroot()

# KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

# Class name mapping (Chinese to integer) - matching K-means clusters for comparison
class_mapping = {
    '森林': 0,
    '水體': 1,
    '裸露地': 2,
    '農地': 3,
    '建物': 4
}

CLASS_NAMES = ['Forest', 'Water', 'Bare Soil', 'Agriculture', 'Built-up']
CLASS_NAMES_ZH = ['森林', '水體', '裸露地', '農地', '建物']

# Extract all Placemarks (bypassing folder hierarchy)
placemarks = root.findall('.//kml:Placemark', ns)

print(f"Found {len(placemarks)} Placemarks in KML")

# Build list of polygons with class labels
polygons_data = []
for pm in placemarks:
    # Get name
    name_elem = pm.find('kml:name', ns)
    if name_elem is None:
        continue
    name = name_elem.text.strip()
    
    # Check if name matches our class mapping
    if name not in class_mapping:
        print(f"  Skipping: '{name}' (not in class mapping)")
        continue
    
    class_id = class_mapping[name]
    
    # Get coordinates from Polygon
    coord_elem = pm.find('.//kml:Polygon//kml:coordinates', ns)
    if coord_elem is None:
        print(f"  Skipping: '{name}' (no coordinates found)")
        continue
    
    # Parse coordinates
    coords_text = coord_elem.text.strip()
    points = []
    for coord in coords_text.split():
        parts = coord.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            points.append((lon, lat))
    
    if len(points) >= 3:
        poly = Polygon(points)
        if poly.is_valid and poly.area > 0:
            polygons_data.append({
                'geometry': poly,
                'class_id': class_id,
                'name': name
            })
            print(f"  ✓ '{name}' → Class {class_id} ({CLASS_NAMES[class_id]})")
        else:
            print(f"  Skipping: '{name}' (invalid polygon)")
    else:
        print(f"  Skipping: '{name}' (insufficient points)")

print(f"\nValid polygons extracted: {len(polygons_data)}")

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(polygons_data, geometry='geometry', crs='EPSG:4326')

print(f"\nGeoDataFrame created:")
print(f"  CRS: {gdf.crs}")
print(f"  Number of polygons: {len(gdf)}")
print(f"  Class distribution:")
for class_id in range(5):
    count = len(gdf[gdf['class_id'] == class_id])
    print(f"    Class {class_id} ({CLASS_NAMES_ZH[class_id]}): {count} polygons")

# =============================================================================
# 2. Reproject and Rasterize Training Polygons
# =============================================================================

print("\nStep 2: Reproject and Rasterize Training Polygons")
print("-" * 60)

# Reproject GeoDataFrame to match image CRS
gdf_reprojected = gdf.to_crs(img_crs)

print(f"Reprojected to: {gdf_reprojected.crs}")

# Prepare shapes for rasterization
shapes = [(row.geometry, row.class_id) for _, row in gdf_reprojected.iterrows()]

# Rasterize polygons
y_mask = rasterize(
    shapes,
    out_shape=(n_rows, n_cols),
    transform=img_transform,
    fill=-1,
    dtype=np.int8,
    all_touched=True
)

print(f"Training mask shape: {y_mask.shape}")
print(f"Training pixels: {np.sum(y_mask != -1)}")

# =============================================================================
# 3. Extract Training Pixels and Split Data
# =============================================================================

print("\nStep 3: Extract Training Pixels and Split Data")
print("-" * 60)

# Reshape image to (n_pixels, n_bands)
X = img.reshape(n_bands, -1).T

# Find valid training pixels (y_mask != -1 AND not NaN in image)
valid_training_mask = (y_mask.flatten() != -1) & ~np.any(np.isnan(X), axis=1)
X_samples = X[valid_training_mask]
y_samples = y_mask.flatten()[valid_training_mask]

print(f"Training samples extracted: {len(y_samples)}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_samples, y_samples,
    test_size=0.2,
    random_state=42,
    stratify=y_samples
)

print(f"Training set: {len(y_train)} samples")
print(f"Test set: {len(y_test)} samples")

# =============================================================================
# 4. Train Random Forest Classifier
# =============================================================================

print("\nStep 4: Train Random Forest Classifier")
print("-" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

# Print accuracies
train_acc = accuracy_score(y_train, rf.predict(X_train))
test_acc = accuracy_score(y_test, rf.predict(X_test))
oob_acc = rf.oob_score_

print(f"\nRandom Forest Performance:")
print(f"  Training Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  OOB Score: {oob_acc:.4f}")

# =============================================================================
# 5. Classify Entire Image
# =============================================================================

print("\nStep 5: Classify Entire Image")
print("-" * 60)

# Get all valid pixels (not NaN)
valid_pixel_mask = ~np.any(np.isnan(X), axis=1)
X_valid = X[valid_pixel_mask]

print(f"Valid pixels for classification: {len(X_valid)}")

# Predict for all valid pixels
print("Predicting land cover for entire image...")
y_pred = rf.predict(X_valid)

# Reconstruct classification map
class_map = np.full(X.shape[0], np.nan)
class_map[valid_pixel_mask] = y_pred
class_map = class_map.reshape(n_rows, n_cols)

# Apply median filter to reduce salt-and-pepper noise
print("Applying median filter...")
class_map_filtered = median_filter(class_map, size=3, mode='nearest')

print(f"Classification map shape: {class_map_filtered.shape}")

# =============================================================================
# 6. Visualize Classification Map
# =============================================================================

print("\nStep 6: Visualize Classification Map")
print("-" * 60)


# 重新對齊顏色：0:森林(綠), 1:水體(藍), 2:裸露地(棕), 3:農地(黃), 4:建物(灰)
colors = ['#228B22', '#0077BE', '#CD853F', '#DAA520', '#808080']

cmap_custom = ListedColormap(colors)
 
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(class_map_filtered, cmap=cmap_custom, vmin=-0.5, vmax=4.5)
cbar = plt.colorbar(im, ax=ax, ticks=range(5), shrink=0.8)
cbar.set_ticklabels(CLASS_NAMES_ZH)
cbar.set_label("Land Cover Class", fontsize=12)
ax.set_title("Random Forest Supervised Classification\nXiulin/Taroko Region", fontsize=14)
ax.set_xlabel("Column")
ax.set_ylabel("Row")
plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "rf_classification.png"), dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 7. Feature Importance
# =============================================================================

print("\nStep 7: Feature Importance")
print("-" * 60)

feature_importance = rf.feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(BAND_NAMES))
ax.barh(y_pos, feature_importance, color='steelblue', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{BAND_NAMES[i]}\n({BANDS[i]})" for i in range(len(BAND_NAMES))])
ax.set_xlabel('Feature Importance', fontsize=13)
ax.set_title('Random Forest Feature Importance\nXiulin/Taroko Region', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "rf_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.show()

# Print feature importance table
print("\nFeature Importance:")
for i, (name, band, importance) in enumerate(zip(BAND_NAMES, BANDS, feature_importance)):
    print(f"  {name:6s} ({band}): {importance:.4f}")

print("\n" + "=" * 60)
print("Task 2 Completed Successfully!")
print("=" * 60)
