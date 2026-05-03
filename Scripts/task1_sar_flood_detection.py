"""
Task 1: SAR All-Weather Flood Detection
========================================
This script performs flood detection using Sentinel-1 RTC data via Planetary Computer STAC API.
It implements the ARIA v7.0 SAR processing pipeline for the 2025 Typhoon Fung-wong flood event.

Physical Logic:
- SAR (Synthetic Aperture Radar) operates at C-band (~5.6 GHz) and can penetrate clouds, making it
  ideal for all-weather flood monitoring.
- Water surfaces act as specular reflectors, returning very low backscatter (typically -25 to -20 dB).
- Land surfaces (vegetation, soil, buildings) return higher backscatter due to diffuse scattering.
- Speckle noise is inherent to coherent SAR imaging due to random interference of scattered waves.
  Median filtering is used because it preserves edges while removing salt-and-pepper noise.
- A relaxed threshold (-14 dB) is used to capture all potential flooded areas, followed by
  morphological cleanup to remove false positives from speckle noise.

Author: Remote Sensing Course Week 10
Date: 2025
"""

import os
import time
import warnings
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening, label
import pystac_client
import planetary_computer as pc
import stackstac

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure matplotlib for Chinese font support (optional, for Taiwan context)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_environment():
    """
    Load environment variables from .env file.
    
    Expected variables:
    - HUALIEN_BBOX: [min_lon, min_lat, max_lon, max_lat]
    - POST_DATE_RANGE: 'YYYY-MM-DD/YYYY-MM-DD'
    - SAR_THRESHOLD: dB threshold for water detection (default: -14)
    - MIN_WATER_PIXELS: minimum pixel count for water objects (default: 50)
    
    Also sets GDAL environment variables for COG streaming timeout handling.
    
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
        'sar_threshold': float(os.getenv('SAR_THRESHOLD', '-14')),
        'min_water_pixels': int(os.getenv('MIN_WATER_PIXELS', '50'))
    }
    
    print('✅ Environment variables loaded:')
    print(f'   BBOX: {env_vars["bbox"]}')
    print(f'   Date Range: {env_vars["date_range"]}')
    print(f'   SAR Threshold: {env_vars["sar_threshold"]} dB')
    print(f'   Min Water Pixels: {env_vars["min_water_pixels"]}')
    print('   GDAL HTTP timeout settings configured')
    
    return env_vars


def search_sentinel1_rtc(catalog, bbox, date_range, max_items=5, max_retries=5):
    """
    Search for Sentinel-1 RTC (Radiometrically Terrain Corrected) data with retry logic.
    
    Sentinel-1 RTC provides gamma-naught backscatter values corrected for
    terrain effects, which is crucial for accurate flood detection in mountainous
    regions like the Matai'an Creek basin.
    
    Args:
        catalog: pystac_client Catalog object
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        date_range: Date range in STAC format 'YYYY-MM-DD/YYYY-MM-DD'
        max_items: Maximum number of items to return (reduced to 5 for faster search)
        max_retries: Maximum number of retry attempts for API timeout
        
    Returns:
        list: List of STAC items
    """
    print(f'\n🔍 Searching Sentinel-1 RTC for {date_range}...')
    print(f'   BBOX: {bbox}')
    print(f'   Max items: {max_items}')
    
    for attempt in range(max_retries):
        try:
            # Use limit parameter to reduce server load
            search = catalog.search(
                collections=['sentinel-1-rtc'],
                bbox=bbox,
                datetime=date_range,
                limit=max_items,
                max_items=max_items,
            )
            
            # Use item_collection() instead of list() for better performance
            items = list(search.item_collection())
            items.sort(key=lambda i: i.properties.get('datetime', ''))
            
            print(f'   Found {len(items)} scenes')
            
            for i, item in enumerate(items):
                dt = item.properties.get('datetime', '?')[:16]
                orbit = item.properties.get('sat:orbit_state', '?')
                print(f'   [{i+1}] {dt} | {orbit} | {item.id[:40]}')
            
            return items
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 + (attempt * 2)  # Longer backoff: 3, 5, 7, 9, 11 seconds
                print(f'   ⚠ API timeout (attempt {attempt + 1}/{max_retries})')
                print(f'   Error: {str(e)[:100]}')
                print(f'   Retrying in {wait_time} seconds...')
                time.sleep(wait_time)
            else:
                raise RuntimeError(f'STAC search failed after {max_retries} attempts: {e}')


def stream_sar_data(item, bands=['vv'], bbox=None, epsg=32651, resolution=10):
    """
    Stream Sentinel-1 data using stackstac.
    
    This function creates a lazy xarray DataArray that can be computed on-demand.
    No files are downloaded to disk - data is streamed directly from COGs.
    
    Args:
        item: STAC item
        bands: List of bands to load (typically 'vv' for VV polarization)
        bbox: Bounding box for the area of interest
        epsg: EPSG code for projection (32651 = UTM Zone 51N for Taiwan)
        resolution: Spatial resolution in meters
        
    Returns:
        xarray.DataArray: Lazy-loaded SAR data
    """
    print(f'\n📡 Streaming SAR data (bands: {bands})...')
    
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
    print(f'   Data type: {cube.dtype}')
    
    return cube


def linear_to_db(linear_data):
    """
    Convert linear backscatter values to decibels (dB).
    
    Physical interpretation:
    - SAR backscatter (sigma-nought) represents the ratio of reflected to incident power
    - dB = 10 * log10(linear_value) provides a logarithmic scale that matches human perception
    - Water typically: -25 to -20 dB (specular reflection)
    - Vegetation: -8 to -3 dB (volume scattering)
    - Urban areas: > 0 dB (double-bounce scattering)
    
    Args:
        linear_data: Linear backscatter values (numpy array or xarray)
        
    Returns:
        numpy.ndarray: Backscatter values in dB
    """
    print('\n🔄 Converting linear to dB...')
    
    # Convert to numpy array if it's an xarray
    if hasattr(linear_data, 'values'):
        linear_values = linear_data.values.squeeze()
    else:
        linear_values = linear_data.squeeze()
    
    # Convert to float32 to avoid overflow and save memory
    linear_values = linear_values.astype(np.float32)
    
    # Apply dB conversion: dB = 10 * log10(linear)
    db_values = 10 * np.log10(linear_values)
    
    # Handle invalid values (NaN, inf, negative)
    db_values = np.where(np.isfinite(db_values), db_values, np.nan)
    
    print(f'   dB range: {np.nanmin(db_values):.1f} ~ {np.nanmax(db_values):.1f} dB')
    
    return db_values


def apply_speckle_filter(db_data, filter_size=5):
    """
    Apply median filter to remove speckle noise.
    
    Speckle noise is a granular noise that inherently exists in SAR images due to
    the coherent nature of radar imaging. It appears as random bright and dark pixels.
    
    Why median filter?
    - Median filter is effective for salt-and-pepper noise while preserving edges
    - It replaces each pixel with the median value of its neighborhood
    - This is crucial BEFORE thresholding to avoid false water detections from dark speckles
    
    Args:
        db_data: SAR data in dB
        filter_size: Size of the median filter kernel (default: 5x5)
        
    Returns:
        numpy.ndarray: Filtered SAR data
    """
    print(f'\n🔧 Applying median filter ({filter_size}x{filter_size}) for speckle removal...')
    
    filtered_data = median_filter(db_data, size=filter_size)
    
    print(f'   Filtered dB range: {np.nanmin(filtered_data):.1f} ~ {np.nanmax(filtered_data):.1f} dB')
    
    return filtered_data


def create_water_mask(filtered_db, threshold):
    """
    Create binary water mask using dB threshold.
    
    Physical basis:
    - Water surfaces are smooth and act as specular reflectors
    - Most radar energy is reflected away from the sensor, resulting in low backscatter
    - Threshold of -14 dB is relaxed to capture all potential flooded areas
    - This will include some false positives, which are removed by morphological cleanup
    
    Args:
        filtered_db: Filtered SAR data in dB
        threshold: dB threshold for water detection
        
    Returns:
        numpy.ndarray: Binary mask (1 = water, 0 = non-water)
    """
    print(f'\n💧 Creating water mask (threshold < {threshold} dB)...')
    
    water_mask = (filtered_db < threshold).astype(np.uint8)
    
    raw_count = np.sum(water_mask)
    print(f'   Raw water pixels: {raw_count:,}')
    
    return water_mask


def morphological_cleanup(water_mask, min_pixels=50):
    """
    Apply morphological operations to clean up the water mask.
    
    Steps:
    1. Binary opening: Erosion followed by dilation
       - Removes small isolated pixels (speckle remnants)
       - Breaks thin connections between separate water bodies
    2. Connected component filtering:
       - Labels connected regions
       - Removes regions smaller than minimum pixel count
       - This eliminates false positives from small noise clusters
    
    Args:
        water_mask: Binary water mask
        min_pixels: Minimum pixel count for valid water objects
        
    Returns:
        numpy.ndarray: Cleaned binary water mask
    """
    print(f'\n🧹 Morphological cleanup (min {min_pixels} pixels)...')
    
    # Step 1: Binary opening (erosion + dilation)
    struct = np.ones((3, 3), dtype=np.uint8)
    opened_mask = binary_opening(water_mask, structure=struct, iterations=1).astype(np.uint8)
    
    opened_count = np.sum(opened_mask)
    removed_opening = np.sum(water_mask) - opened_count
    print(f'   After opening: {opened_count:,} px (removed {removed_opening:,} px)')
    
    # Step 2: Connected component filtering
    labeled, n_features = label(opened_mask)
    
    cleaned_mask = np.zeros_like(opened_mask)
    kept_regions = 0
    removed_regions = 0
    
    for region_id in range(1, n_features + 1):
        region_size = np.sum(labeled == region_id)
        if region_size >= min_pixels:
            cleaned_mask[labeled == region_id] = 1
            kept_regions += 1
        else:
            removed_regions += 1
    
    final_count = np.sum(cleaned_mask)
    print(f'   Connected components: {n_features} regions')
    print(f'   Kept {kept_regions} regions (≥ {min_pixels} px)')
    print(f'   Removed {removed_regions} small regions')
    print(f'   Final water pixels: {final_count:,}')
    
    return cleaned_mask


def calculate_flood_area(water_mask, pixel_size_m=10):
    """
    Calculate flooded area in square kilometers.
    
    Args:
        water_mask: Binary water mask
        pixel_size_m: Pixel size in meters (default: 10m for Sentinel-1)
        
    Returns:
        float: Flooded area in km²
    """
    water_pixels = np.sum(water_mask)
    pixel_area_m2 = pixel_size_m * pixel_size_m
    total_area_m2 = water_pixels * pixel_area_m2
    total_area_km2 = total_area_m2 / 1_000_000  # Convert m² to km²
    
    return total_area_km2


def visualize_results(raw_db, filtered_db, water_mask, threshold, output_path='output/task1_sar_results.png'):
    """
    Create a 2x2 subplot visualization of SAR processing results.
    
    Subplots:
    (a) Raw SAR (dB) - Shows original backscatter values
    (b) Filtered SAR (dB) - Shows speckle-filtered data
    (c) Binary flood mask - Shows detected water bodies
    (d) Histogram - Shows distribution of dB values with threshold line
    
    Args:
        raw_db: Raw SAR data in dB
        filtered_db: Filtered SAR data in dB
        water_mask: Binary water mask
        threshold: dB threshold used for water detection
        output_path: Path to save the figure
    """
    print(f'\n📊 Creating visualization...')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # (a) Raw SAR
    im0 = axes[0, 0].imshow(raw_db, cmap='gray', vmin=-30, vmax=0)
    axes[0, 0].set_title('(a) Raw SAR VV (dB)', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], label='Backscatter (dB)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    
    # (b) Filtered SAR
    im1 = axes[0, 1].imshow(filtered_db, cmap='gray', vmin=-30, vmax=0)
    axes[0, 1].set_title('(b) Filtered SAR (Median 5×5)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], label='Backscatter (dB)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    
    # (c) Binary flood mask
    im2 = axes[1, 0].imshow(water_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title(f'(c) Flood Mask (VV < {threshold} dB)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0], label='Water (1) / Land (0)')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    
    # (d) Histogram
    valid_pixels = filtered_db[~np.isnan(filtered_db)]
    axes[1, 1].hist(valid_pixels, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold} dB')
    axes[1, 1].set_title('(d) SAR Backscatter Histogram', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Backscatter (dB)')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Task 1: SAR All-Weather Flood Detection\nARIA v7.0 Processing Pipeline',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'   Figure saved to: {output_path}')
    
    plt.show()


def main():
    """
    Main execution function for SAR flood detection pipeline.
    """
    print('=' * 70)
    print('Task 1: SAR All-Weather Flood Detection')
    print('ARIA v7.0 Processing Pipeline')
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
    
    # Step 3: Search for Sentinel-1 RTC data
    items = search_sentinel1_rtc(
        catalog,
        bbox=env['bbox'],
        date_range=env['date_range']
    )
    
    if not items:
        raise ValueError('No Sentinel-1 RTC scenes found for the specified parameters.')
    
    # Step 4: Select first item and print metadata
    selected_item = items[0]
    print(f'\n✅ Selected scene:')
    print(f'   Date: {selected_item.properties["datetime"][:10]}')
    print(f'   Orbit: {selected_item.properties.get("sat:orbit_state", "unknown")}')
    print(f'   ID: {selected_item.id}')
    
    # Step 5: Stream SAR data
    sar_lazy = stream_sar_data(
        selected_item,
        bands=['vv'],
        bbox=env['bbox'],
        epsg=32651,
        resolution=10
    )
    
    # Step 6: Compute the lazy array (this triggers the actual data download)
    print('\n⏳ Computing SAR data (this may take a moment)...')
    sar_linear = sar_lazy.compute()
    print('✅ SAR data computed')
    
    # Step 7: Convert to dB
    sar_db = linear_to_db(sar_linear)
    raw_db = sar_db.copy()  # Keep copy for visualization
    
    # Step 8: Apply speckle filter
    sar_filtered = apply_speckle_filter(sar_db, filter_size=5)
    
    # Step 9: Create water mask
    water_mask_raw = create_water_mask(sar_filtered, env['sar_threshold'])
    
    # Step 10: Morphological cleanup
    water_mask_clean = morphological_cleanup(
        water_mask_raw,
        min_pixels=env['min_water_pixels']
    )
    
    # Step 11: Calculate flooded area
    flood_area_km2 = calculate_flood_area(water_mask_clean, pixel_size_m=10)
    print(f'\n📏 Flooded Area: {flood_area_km2:.2f} km²')
    
    # Step 12: Visualization
    visualize_results(
        raw_db,
        sar_filtered,
        water_mask_clean,
        env['sar_threshold'],
        output_path='output/task1_sar_results.png'
    )
    
    print('\n' + '=' * 70)
    print('✅ Task 1 Complete: SAR All-Weather Flood Detection')
    print('=' * 70)


if __name__ == '__main__':
    main()
