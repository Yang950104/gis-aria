"""
Task 2: Sensor Fusion — Multi-Source Confidence Map
====================================================
This script combines SAR flood detection with optical Sentinel-2 data to create
a 4-class confidence map for the ARIA v7.0 All-Weather Auditor.

Physical Logic:
- SAR provides all-weather flood detection but can have false positives from
  radar shadow and layover effects in mountainous terrain.
- Optical data (Sentinel-2) provides high-confidence water detection via NDWI
  but is blocked by clouds.
- By fusing both sensors, we can:
  1. High Confidence (Class 3): Both SAR and optical agree on water presence
  2. SAR Only (Class 2): SAR detects water in cloudy areas where optical fails
  3. Optical Only (Class 1): Optical detects water that SAR missed (rare)
  4. No Detection (Class 0): Neither sensor detects water

SCL (Scene Classification Layer) Cloud Mask:
- SCL values 3, 8, 9, 10 represent clouds, cloud shadows, and cirrus
- These pixels must be masked to avoid false positives from cloud shadows
  being misclassified as water by NDWI
- This is critical because cloud shadows can have low reflectance similar to water

Author: Remote Sensing Course Week 10
Date: 2025
"""

import os
import time
import warnings
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
import pystac_client
import planetary_computer as pc
import stackstac

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_environment():
    """
    Load environment variables from .env file.
    
    Expected variables:
    - HUALIEN_BBOX: [min_lon, min_lat, max_lon, max_lat]
    - POST_DATE_RANGE: 'YYYY-MM-DD/YYYY-MM-DD'
    - NDWI_THRESHOLD: NDWI threshold for optical water detection (default: 0.3)
    
    Also sets GDAL environment variables for COG streaming.
    
    Returns:
        dict: Dictionary containing environment variables
    """
    load_dotenv()
    
    # Set GDAL environment variables for COG streaming
    os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '5')
    os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
    os.environ.setdefault('GDAL_HTTP_TIMEOUT', '60')
    os.environ.setdefault('GDAL_HTTP_MULTIRANGE', 'YES')
    os.environ.setdefault('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    os.environ.setdefault('VSI_CACHE', 'TRUE')
    os.environ.setdefault('VSI_CACHE_SIZE', '1000000000')
    os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.TIF,.tiff')
    
    env_vars = {
        'bbox': eval(os.getenv('HUALIEN_BBOX', '[121.270, 23.685, 121.310, 23.715]')),
        'date_range': os.getenv('POST_DATE_RANGE', '2025-09-10/2025-09-25'),
        'ndwi_threshold': float(os.getenv('NDWI_THRESHOLD', '0.3'))
    }
    
    print('✅ Environment variables loaded:')
    print(f'   BBOX: {env_vars["bbox"]}')
    print(f'   Date Range: {env_vars["date_range"]}')
    print(f'   NDWI Threshold: {env_vars["ndwi_threshold"]}')
    
    return env_vars


def search_sentinel2_l2a(catalog, bbox, date_range, max_items=20, max_retries=3):
    """
    Search for Sentinel-2 L2A data and select scene with lowest cloud cover.
    
    Sentinel-2 L2A provides Bottom-of-Atmosphere reflectance values with
    Scene Classification Layer (SCL) for cloud detection.
    
    Args:
        catalog: pystac_client Catalog object
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        date_range: Date range in STAC format 'YYYY-MM-DD/YYYY-MM-DD'
        max_items: Maximum number of items to return
        max_retries: Maximum number of retry attempts for API timeout
        
    Returns:
        dict: Selected STAC item with lowest cloud cover
    """
    print(f'\n🔍 Searching Sentinel-2 L2A for {date_range}...')
    
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=['sentinel-2-l2a'],
                bbox=bbox,
                datetime=date_range,
                max_items=max_items,
            )
            
            items = list(search.items())
            
            if not items:
                raise ValueError('No Sentinel-2 L2A scenes found.')
            
            # Sort by cloud cover (ascending)
            items.sort(key=lambda i: i.properties.get('eo:cloud_cover', 100))
            
            # Select item with lowest cloud cover
            selected_item = items[0]
            cloud_cover = selected_item.properties.get('eo:cloud_cover', -1)
            
            print(f'   Found {len(items)} scenes')
            print(f'   Selected: {selected_item.properties["datetime"][:10]}')
            print(f'   Cloud cover: {cloud_cover:.1f}%')
            
            # Print top 3 scenes for reference
            print(f'   Top 3 scenes by cloud cover:')
            for i, item in enumerate(items[:3]):
                cc = item.properties.get('eo:cloud_cover', -1)
                dt = item.properties.get('datetime', '?')[:10]
                print(f'     [{i+1}] {dt} | {cc:.1f}% cloud')
            
            return selected_item
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f'   ⚠ API timeout (attempt {attempt + 1}/{max_retries})')
                print(f'   Retrying in {wait_time} seconds...')
                time.sleep(wait_time)
            else:
                raise RuntimeError(f'STAC search failed after {max_retries} attempts: {e}')


def stream_optical_data(item, bands=['B03', 'B08', 'SCL'], bbox=None, epsg=32651, resolution=10):
    """
    Stream Sentinel-2 optical data using stackstac.
    
    Bands:
    - B03: Green (560 nm center wavelength)
    - B08: NIR (842 nm center wavelength)
    - SCL: Scene Classification Layer (20m resolution, will be resampled)
    
    Args:
        item: STAC item
        bands: List of bands to load
        bbox: Bounding box for the area of interest
        epsg: EPSG code for projection
        resolution: Spatial resolution in meters
        
    Returns:
        xarray.Dataset: Lazy-loaded optical data
    """
    print(f'\n📡 Streaming optical data (bands: {bands})...')
    
    signed = pc.sign(item)
    
    cube = stackstac.stack(
        [signed],
        assets=bands,
        epsg=epsg,
        resolution=resolution,
        bounds_latlon=bbox,
        chunksize=2048,
    ).squeeze('time')
    
    print(f'   Data shape: {cube.shape}')
    print(f'   Bands: {list(cube.band.values)}')
    
    return cube


def compute_ndwi(green, nir):
    """
    Compute Normalized Difference Water Index (NDWI).
    
    Physical basis:
    - Water has strong absorption in NIR and moderate reflectance in Green
    - NDWI = (Green - NIR) / (Green + NIR)
    - Water typically has NDWI > 0.3
    - Vegetation has NDWI < 0 (high NIR reflectance)
    - Built-up areas have NDWI around 0
    
    Args:
        green: Green band reflectance values
        nir: NIR band reflectance values
        
    Returns:
        numpy.ndarray: NDWI values
    """
    print('\n💧 Computing NDWI...')
    
    # Avoid division by zero
    ndwi = (green - nir) / (green + nir + 1e-9)
    
    print(f'   NDWI range: {np.nanmin(ndwi):.3f} ~ {np.nanmax(ndwi):.3f}')
    
    return ndwi


def create_optical_water_mask(ndwi, threshold):
    """
    Create optical water mask using NDWI threshold.
    
    Args:
        ndwi: NDWI values
        threshold: NDWI threshold for water detection
        
    Returns:
        numpy.ndarray: Binary mask (1 = water, 0 = non-water)
    """
    print(f'\n💧 Creating optical water mask (NDWI > {threshold})...')
    
    water_mask = (ndwi > threshold).astype(np.uint8)
    
    water_pixels = np.sum(water_mask)
    print(f'   Optical water pixels: {water_pixels:,}')
    
    return water_mask


def create_cloud_mask(scl):
    """
    Create cloud mask from SCL (Scene Classification Layer).
    
    SCL values:
    - 0: No data
    - 1: Saturated/defective
    - 2: Dark area pixels
    - 3: Cloud shadows
    - 4: Vegetation
    - 5: Bare soils
    - 6: Water
    - 7: Clouds low probability / unclassified
    - 8: Clouds medium probability
    - 9: Clouds high probability
    - 10: Thin cirrus
    
    Cloud/shadow values to mask: 3, 8, 9, 10
    
    Why this is critical:
    - Cloud shadows have low reflectance similar to water
    - Without proper cloud masking, shadows would be falsely classified as water by NDWI
    - This is a major source of false positives in optical flood detection
    
    Args:
        scl: SCL band values
        
    Returns:
        numpy.ndarray: Binary mask (1 = clear, 0 = cloud/shadow)
    """
    print('\n☁️ Creating cloud mask from SCL...')
    
    # Cloud and shadow values: 3 (shadow), 8 (cloud med), 9 (cloud high), 10 (cirrus)
    cloud_values = [3, 8, 9, 10]
    
    # cloud_mask = 1 for clear pixels, 0 for cloudy/shadowed pixels
    cloud_mask = (~np.isin(scl, cloud_values)).astype(np.uint8)
    
    clear_pixels = np.sum(cloud_mask)
    total_pixels = scl.size
    cloud_percentage = (1 - clear_pixels / total_pixels) * 100
    
    print(f'   Clear pixels: {clear_pixels:,} / {total_pixels:,} ({100-cloud_percentage:.1f}%)')
    print(f'   Cloud/shadow pixels: {total_pixels - clear_pixels:,} ({cloud_percentage:.1f}%)')
    
    return cloud_mask


def align_grids(sar_mask, optical_mask):
    """
    Align SAR and optical masks to the same grid dimensions.
    
    Crucial step because:
    - S1 and S2 may have slight shape differences despite same bbox and resolution
    - This can be due to different grid origins or rounding in projection
    - We clip to the minimum common dimensions to ensure pixel-wise alignment
    
    Args:
        sar_mask: SAR water mask
        optical_mask: Optical water mask or cloud mask
        
    Returns:
        tuple: (aligned_sar_mask, aligned_optical_mask)
    """
    print('\n🔧 Aligning SAR and optical grids...')
    
    sar_shape = sar_mask.shape
    opt_shape = optical_mask.shape
    
    print(f'   SAR shape: {sar_shape}')
    print(f'   Optical shape: {opt_shape}')
    
    # Find minimum common dimensions
    min_h = min(sar_shape[0], opt_shape[0])
    min_w = min(sar_shape[1], opt_shape[1])
    
    # Clip both arrays to minimum dimensions
    sar_aligned = sar_mask[:min_h, :min_w]
    opt_aligned = optical_mask[:min_h, :min_w]
    
    print(f'   Aligned shape: {sar_aligned.shape}')
    
    return sar_aligned, opt_aligned


def create_confidence_map(sar_water, optical_water, cloud_mask):
    """
    Create 4-class confidence map based on fusion logic.
    
    Confidence Matrix:
    - Class 3 (High Confidence): Optical Water=True AND SAR Water=True AND Cloud=False
      Both sensors agree on water presence in clear conditions
    - Class 2 (SAR Only - Cloudy): SAR Water=True AND Cloud=True
      SAR detects water in cloudy areas where optical fails
    - Class 1 (Optical Only): Optical Water=True AND SAR Water=False AND Cloud=False
      Optical detects water that SAR missed (rare, e.g., very calm water)
    - Class 0 (No Detection): Everything else
      Neither sensor detects water
    
    Args:
        sar_water: Binary SAR water mask
        optical_water: Binary optical water mask
        cloud_mask: Binary cloud mask (1 = clear, 0 = cloud)
        
    Returns:
        numpy.ndarray: 4-class confidence map (0, 1, 2, 3)
    """
    print('\n🎯 Creating confidence map...')
    
    # Initialize confidence map with Class 0 (No Detection)
    confidence_map = np.zeros_like(sar_water, dtype=np.uint8)
    
    # Class 3: High Confidence (both sensors agree, clear sky)
    confidence_map[(optical_water == 1) & (sar_water == 1) & (cloud_mask == 1)] = 3
    
    # Class 2: SAR Only (SAR detects water in cloudy areas)
    confidence_map[(sar_water == 1) & (cloud_mask == 0) & (confidence_map != 3)] = 2
    
    # Class 1: Optical Only (optical detects water, SAR doesn't, clear sky)
    confidence_map[(optical_water == 1) & (sar_water == 0) & (cloud_mask == 1)] = 1
    
    # Class 0: No Detection (already initialized)
    
    # Print statistics
    for class_id in range(4):
        count = np.sum(confidence_map == class_id)
        percentage = count / confidence_map.size * 100
        print(f'   Class {class_id}: {count:,} px ({percentage:.1f}%)')
    
    return confidence_map


def calculate_class_areas(confidence_map, pixel_size_m=10):
    """
    Calculate area in km² for each confidence class.
    
    Args:
        confidence_map: 4-class confidence map
        pixel_size_m: Pixel size in meters
        
    Returns:
        dict: Dictionary with area for each class
    """
    print('\n📏 Calculating class areas...')
    
    pixel_area_m2 = pixel_size_m * pixel_size_m
    pixel_area_km2 = pixel_area_m2 / 1_000_000
    
    areas = {}
    for class_id in range(4):
        count = np.sum(confidence_map == class_id)
        area_km2 = count * pixel_area_km2
        areas[class_id] = area_km2
    
    return areas


def visualize_confidence_map(confidence_map, areas, output_path='output/task2_confidence_map.png'):
    """
    Visualize the 4-class confidence map with custom colormap.
    
    Colormap:
    - Class 0 (No Detection): Light gray (#E8E8E8)
    - Class 1 (Optical Only): Green (#3182CE)
    - Class 2 (SAR Only): Blue (#FF6B2A)
    - Class 3 (High Confidence): Red (#D24817)
    
    Args:
        confidence_map: 4-class confidence map
        areas: Dictionary with area for each class
        output_path: Path to save the figure
    """
    print(f'\n📊 Creating confidence map visualization...')
    
    # Custom colormap
    colors = ['#E8E8E8', '#3182CE', '#FF6B2A', '#D24817']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(confidence_map, cmap=cmap, norm=norm)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Class 0: No Detection'),
        Patch(facecolor=colors[1], label=f'Class 1: Optical Only ({areas[1]:.2f} km²)'),
        Patch(facecolor=colors[2], label=f'Class 2: SAR Only - Cloudy ({areas[2]:.2f} km²)'),
        Patch(facecolor=colors[3], label=f'Class 3: High Confidence ({areas[3]:.2f} km²)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('Task 2: Sensor Fusion — Multi-Source Confidence Map\nARIA v7.0 All-Weather Auditor',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'   Figure saved to: {output_path}')
    
    plt.show()


def main():
    """
    Main execution function for sensor fusion pipeline.
    """
    print('=' * 70)
    print('Task 2: Sensor Fusion — Multi-Source Confidence Map')
    print('ARIA v7.0 All-Weather Auditor')
    print('=' * 70)
    
    # Step 1: Load environment variables
    env = load_environment()
    
    # Step 2: Connect to Planetary Computer STAC API
    print('\n🌐 Connecting to Planetary Computer STAC API...')
    catalog = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=pc.sign_inplace,
    )
    print('✅ Connected to STAC API')
    
    # Step 3: Search for Sentinel-2 L2A data
    s2_item = search_sentinel2_l2a(
        catalog,
        bbox=env['bbox'],
        date_range=env['date_range']
    )
    
    # Step 4: Stream optical data
    optical_lazy = stream_optical_data(
        s2_item,
        bands=['B03', 'B08', 'SCL'],
        bbox=env['bbox'],
        epsg=32651,
        resolution=10
    )
    
    # Step 5: Compute optical data
    print('\n⏳ Computing optical data...')
    optical_data = optical_lazy.compute()
    print('✅ Optical data computed')
    
    # Extract bands
    green = optical_data.sel(band='B03').values.squeeze()
    nir = optical_data.sel(band='B08').values.squeeze()
    scl = optical_data.sel(band='SCL').values.squeeze()
    
    # Step 6: Compute NDWI
    ndwi = compute_ndwi(green, nir)
    
    # Step 7: Create optical water mask
    optical_water = create_optical_water_mask(ndwi, env['ndwi_threshold'])
    
    # Step 8: Create cloud mask from SCL
    cloud_mask = create_cloud_mask(scl)
    
    # Step 9: Load SAR water mask (assume it was saved from Task 1)
    # For this example, we'll need to either:
    # a) Load from a file saved by Task 1, or
    # b) Re-run the SAR detection pipeline
    
    # For now, let's assume we need to re-run SAR detection
    print('\n⚠ SAR water mask not found. Running SAR detection pipeline...')
    
    # Import SAR detection functions from Task 1
    # (In practice, you would import from task1_sar_flood_detection)
    # For this standalone script, we'll implement a simplified version
    
    from task1_sar_flood_detection import (
        search_sentinel1_rtc, stream_sar_data, linear_to_db,
        apply_speckle_filter, create_water_mask, morphological_cleanup
    )
    
    # Search SAR data
    sar_items = search_sentinel1_rtc(catalog, env['bbox'], env['date_range'])
    sar_item = sar_items[0]
    
    # Stream and process SAR
    sar_lazy = stream_sar_data(sar_item, ['vv'], env['bbox'])
    sar_linear = sar_lazy.compute()
    sar_db = linear_to_db(sar_linear)
    sar_filtered = apply_speckle_filter(sar_db, filter_size=5)
    
    # Use default SAR threshold of -14 dB if not in env
    sar_threshold = float(os.getenv('SAR_THRESHOLD', '-14'))
    min_water_pixels = int(os.getenv('MIN_WATER_PIXELS', '50'))
    
    sar_water_raw = create_water_mask(sar_filtered, sar_threshold)
    sar_water = morphological_cleanup(sar_water_raw, min_water_pixels)
    
    # Step 10: Align grids
    sar_water_aligned, optical_water_aligned = align_grids(sar_water, optical_water)
    _, cloud_mask_aligned = align_grids(sar_water, cloud_mask)
    
    # Step 11: Create confidence map
    confidence_map = create_confidence_map(
        sar_water_aligned,
        optical_water_aligned,
        cloud_mask_aligned
    )
    
    # Step 12: Calculate areas
    areas = calculate_class_areas(confidence_map, pixel_size_m=10)
    
    print('\n📊 Area Statistics:')
    print(f'   Class 0 (No Detection): {areas[0]:.2f} km²')
    print(f'   Class 1 (Optical Only): {areas[1]:.2f} km²')
    print(f'   Class 2 (SAR Only - Cloudy): {areas[2]:.2f} km²')
    print(f'   Class 3 (High Confidence): {areas[3]:.2f} km²')
    print(f'   Total detected water: {areas[1] + areas[2] + areas[3]:.2f} km²')
    
    # Step 13: Visualization
    visualize_confidence_map(confidence_map, areas, output_path='output/task2_confidence_map.png')
    
    print('\n' + '=' * 70)
    print('✅ Task 2 Complete: Sensor Fusion — Multi-Source Confidence Map')
    print('=' * 70)


if __name__ == '__main__':
    main()
