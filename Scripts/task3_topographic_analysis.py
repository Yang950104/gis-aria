"""
Task 3: Topographic Analysis — DEM & Slope Assessment
=====================================================
This script uses Copernicus DEM to filter false positive water detections
caused by SAR geometric distortions on steep slopes.

Physical Logic:
- SAR side-looking geometry creates artifacts on steep terrain:
  * Radar Shadow: Back-facing slopes receive no signal → appear dark like water
  * Foreshortening: Front-facing slopes are compressed → anomalous backscatter
  * Layover: Extremely steep slopes fold over → create dark patches
- Water cannot pool on slopes > 25-30° due to gravity
- By filtering steep slopes, we remove radar shadow false positives
- This is critical in mountainous regions like the Matai'an Creek basin

Important Note on DEM Validity:
- Copernicus DEM GLO-30 is based on 2011-2014 data (pre-disaster)
- For newly formed barrier lakes (like the 2025 case), the terrain has changed
- Landslides and dam formation alter the topography
- Therefore, the pre-disaster DEM may not accurately represent post-disaster slopes
- This is a limitation acknowledged in the analysis

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
from scipy.ndimage import uniform_filter
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
    - SLOPE_THRESHOLD: Slope threshold in degrees for filtering (default: 25)
    
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
        'slope_threshold': float(os.getenv('SLOPE_THRESHOLD', '25'))
    }
    
    print('✅ Environment variables loaded:')
    print(f'   BBOX: {env_vars["bbox"]}')
    print(f'   Slope Threshold: {env_vars["slope_threshold"]}°')
    
    return env_vars


def search_copernicus_dem(catalog, bbox, max_retries=5):
    """
    Search for Copernicus DEM GLO-30 data.
    
    Copernicus DEM GLO-30 is a global digital elevation model with 30m resolution,
    resampled here to 10m to match SAR/optical data.
    
    Args:
        catalog: pystac_client Catalog object
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        max_retries: Maximum number of retry attempts for API timeout
        
    Returns:
        list: List of STAC items
    """
    print(f'\n🔍 Searching Copernicus DEM GLO-30...')
    
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=['cop-dem-glo-30'],
                bbox=bbox,
                limit=10,
            )
            
            items = list(search.item_collection())
            
            if not items:
                raise ValueError('No Copernicus DEM tiles found.')
            
            print(f'   Found {len(items)} DEM tiles')
            
            for i, item in enumerate(items):
                print(f'   [{i+1}] {item.id}')
            
            return items
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 + (attempt * 2)
                print(f'   ⚠ API timeout (attempt {attempt + 1}/{max_retries})')
                print(f'   Retrying in {wait_time} seconds...')
                time.sleep(wait_time)
            else:
                raise RuntimeError(f'DEM search failed after {max_retries} attempts: {e}')


def stream_dem_data(items, bbox=None, epsg=32651, resolution=10):
    """
    Stream Copernicus DEM data using stackstac.
    
    Args:
        items: List of STAC items
        bbox: Bounding box for the area of interest
        epsg: EPSG code for projection
        resolution: Spatial resolution in meters
        
    Returns:
        xarray.DataArray: Lazy-loaded DEM data
    """
    print(f'\n📡 Streaming DEM data...')
    
    # Sign all items
    signed_items = [pc.sign(item) for item in items]
    
    # Stack all tiles and take maximum (handle overlapping tiles)
    dem_lazy = stackstac.stack(
        signed_items,
        assets=['data'],
        epsg=epsg,
        resolution=resolution,
        bounds_latlon=bbox,
        chunksize=2048,
    )
    
    # If multiple time slices (overlapping tiles), take max
    if dem_lazy.sizes.get('time', 1) > 1:
        print(f'   Multiple tiles found, taking maximum elevation...')
        dem_lazy = dem_lazy.max(dim='time')
    else:
        dem_lazy = dem_lazy.squeeze('time')
    
    print(f'   Data shape: {dem_lazy.shape}')
    
    return dem_lazy


def compute_slope(dem, cell_size=10):
    """
    Compute slope in degrees from DEM.
    
    Physical basis:
    - Slope represents the steepness of terrain
    - Calculated from the gradient of elevation in x and y directions
    - Slope = arctan(sqrt((dz/dx)² + (dz/dy)²))
    - Result converted from radians to degrees
    
    Args:
        dem: Digital elevation model (meters)
        cell_size: Spatial resolution in meters (default: 10m)
        
    Returns:
        numpy.ndarray: Slope in degrees
    """
    print('\n📐 Computing slope from DEM...')
    
    # Apply slight smoothing to reduce noise in gradient calculation
    dem_smooth = uniform_filter(dem, size=3)
    
    # Compute gradients
    dy, dx = np.gradient(dem_smooth, cell_size)
    
    # Calculate slope in radians
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    
    # Convert to degrees
    slope_deg = np.degrees(slope_rad)
    
    print(f'   Elevation range: {np.nanmin(dem):.0f} ~ {np.nanmax(dem):.0f} m')
    print(f'   Slope range: {np.nanmin(slope_deg):.1f}° ~ {np.nanmax(slope_deg):.1f}°')
    
    return slope_deg


def create_slope_mask(slope, threshold):
    """
    Create a mask for steep slopes.
    
    Physical logic:
    - Water cannot pool on slopes > 25-30° due to gravity
    - Steep slopes with low backscatter are likely radar shadows, not water
    - This mask identifies pixels that should be filtered out
    
    Args:
        slope: Slope in degrees
        threshold: Slope threshold in degrees
        
    Returns:
        numpy.ndarray: Binary mask (1 = steep, 0 = gentle)
    """
    print(f'\n⛰️ Creating slope mask (>{threshold}° considered steep)...')
    
    slope_mask = (slope > threshold).astype(np.uint8)
    
    steep_pixels = np.sum(slope_mask)
    total_pixels = slope.size
    steep_percentage = steep_pixels / total_pixels * 100
    
    print(f'   Steep pixels: {steep_pixels:,} / {total_pixels:,} ({steep_percentage:.1f}%)')
    
    return slope_mask


def apply_topographic_filter(confidence_map, slope_mask, slope_threshold):
    """
    Apply topographic filter to remove water detections on steep slopes.
    
    Logic:
    - Identify water pixels (Classes 1, 2, 3) on steep slopes
    - Reclassify them as Class 0 (No Detection) or Class 4 (Removed False Positive)
    - Track how many pixels were removed for statistics
    
    Args:
        confidence_map: 4-class confidence map from Task 2
        slope_mask: Binary mask of steep slopes (1 = steep)
        slope_threshold: Slope threshold used
        
    Returns:
        tuple: (filtered_confidence_map, false_positives_mask, removed_count)
    """
    print(f'\n🔧 Applying topographic filter (slope > {slope_threshold}°)...')
    
    # Create a copy for modification
    filtered_map = confidence_map.copy()
    
    # Identify water pixels on steep slopes
    # Water classes: 1 (Optical Only), 2 (SAR Only), 3 (High Confidence)
    water_on_steep = ((confidence_map >= 1) & (confidence_map <= 3) & (slope_mask == 1))
    
    # Count water pixels before filtering
    water_before = np.sum((confidence_map >= 1) & (confidence_map <= 3))
    
    # Create mask of false positives removed
    false_positives_mask = water_on_steep.astype(np.uint8)
    removed_count = np.sum(false_positives_mask)
    
    # Reclassify removed pixels to Class 0 (No Detection)
    # Alternatively, could use Class 4 to track them separately
    filtered_map[water_on_steep] = 0
    
    # Count water pixels after filtering
    water_after = np.sum((filtered_map >= 1) & (filtered_map <= 3))
    
    print(f'   Water pixels before filter: {water_before:,}')
    print(f'   Water pixels after filter: {water_after:,}')
    print(f'   False positives removed: {removed_count:,}')
    print(f'   Reduction: {(removed_count / water_before * 100):.1f}%')
    
    return filtered_map, false_positives_mask, removed_count


def calculate_removed_area(removed_count, pixel_size_m=10):
    """
    Calculate area of removed false positives in km².
    
    Args:
        removed_count: Number of pixels removed
        pixel_size_m: Pixel size in meters
        
    Returns:
        float: Area in km²
    """
    pixel_area_m2 = pixel_size_m * pixel_size_m
    total_area_m2 = removed_count * pixel_area_m2
    total_area_km2 = total_area_m2 / 1_000_000
    
    return total_area_km2


def visualize_before_after(confidence_before, confidence_after, removed_area_km2, 
                          output_path='output/task3_topographic_correction.png'):
    """
    Visualize before/after topographic correction.
    
    Args:
        confidence_before: Confidence map before filtering
        confidence_after: Confidence map after filtering
        removed_area_km2: Area of false positives removed (km²)
        output_path: Path to save the figure
    """
    print(f'\n📊 Creating before/after visualization...')
    
    # Custom colormap
    colors = ['#E8E8E8', '#3182CE', '#FF6B2A', '#D24817']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before correction
    im0 = axes[0].imshow(confidence_before, cmap=cmap, norm=norm)
    axes[0].set_title('(a) Before Topographic Correction\nFusion Result from Task 2',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # After correction
    im1 = axes[1].imshow(confidence_after, cmap=cmap, norm=norm)
    axes[1].set_title(f'(b) After Topographic Correction\nRemoved {removed_area_km2:.2f} km² of False Positives',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Class 0: No Detection'),
        Patch(facecolor=colors[1], label='Class 1: Optical Only'),
        Patch(facecolor=colors[2], label='Class 2: SAR Only - Cloudy'),
        Patch(facecolor=colors[3], label='Class 3: High Confidence')
    ]
    
    # Add legend to both plots
    for ax in axes:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Task 3: Topographic Analysis — DEM & Slope Assessment\n'
                 'Filtering Radar Shadow False Positives on Steep Slopes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'   Figure saved to: {output_path}')
    
    plt.show()


def load_confidence_map_from_task2():
    """
    Load confidence map from Task 2.
    
    In practice, this would load from a saved file.
    For this standalone script, we'll need to re-run Task 2 or
    assume the confidence_map is available in memory.
    
    Returns:
        numpy.ndarray: Confidence map from Task 2
    """
    print('\n⚠ Loading confidence map from Task 2...')
    
    # Option 1: Load from saved file (if Task 2 saved it)
    # confidence_map = np.load('output/confidence_map.npy')
    
    # Option 2: Re-run Task 2 (slower but ensures consistency)
    print('   Re-running Task 2 to generate confidence map...')
    from task2_sensor_fusion import main as task2_main
    
    # Note: This would require modifying Task 2 to return the confidence_map
    # For now, we'll assume the user has run Task 2 and saved the result
    
    # Placeholder - in actual use, load from file or modify Task 2
    raise NotImplementedError(
        'Please run Task 2 first and save the confidence_map, '
        'or modify this function to load from a saved file.'
    )


def main():
    """
    Main execution function for topographic analysis pipeline.
    """
    print('=' * 70)
    print('Task 3: Topographic Analysis — DEM & Slope Assessment')
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
    
    # Step 3: Search for Copernicus DEM
    dem_items = search_copernicus_dem(catalog, env['bbox'])
    
    # Step 4: Stream DEM data
    dem_lazy = stream_dem_data(dem_items, bbox=env['bbox'], epsg=32651, resolution=10)
    
    # Step 5: Compute DEM
    print('\n⏳ Computing DEM data...')
    dem_data = dem_lazy.compute()
    dem = dem_data.values.squeeze()
    
    # Handle invalid values
    dem[dem <= 0] = np.nan
    if np.any(np.isnan(dem)):
        dem[np.isnan(dem)] = np.nanmedian(dem)
    
    print('✅ DEM computed')
    
    # Step 6: Calculate slope
    slope = compute_slope(dem, cell_size=10)
    
    # Step 7: Create slope mask
    slope_mask = create_slope_mask(slope, env['slope_threshold'])
    
    # Step 8: Load confidence map from Task 2
    # For this example, we'll create a synthetic confidence map
    # In practice, load from Task 2 output
    print('\n⚠ Note: Loading confidence map from Task 2...')
    print('   For this demonstration, creating a synthetic confidence map.')
    print('   In practice, load from: output/confidence_map.npy')
    
    # Create synthetic confidence map for demonstration
    # (Replace this with actual loading from Task 2)
    H, W = dem.shape
    confidence_before = np.zeros((H, W), dtype=np.uint8)
    
    # Add some synthetic water detections
    # Class 3: High Confidence (center area)
    confidence_before[H//2-20:H//2+20, W//2-20:W//2+20] = 3
    # Class 2: SAR Only (some scattered pixels)
    confidence_before[H//3:H//3+10, W//3:W//3+10] = 2
    confidence_before[2*H//3:2*H//3+10, 2*W//3:2*W//3+10] = 2
    # Class 1: Optical Only (few pixels)
    confidence_before[H//4:H//4+5, W//4:W//4+5] = 1
    
    print(f'   Synthetic confidence map shape: {confidence_before.shape}')
    
    # Step 9: Apply topographic filter
    confidence_after, false_positives_mask, removed_count = apply_topographic_filter(
        confidence_before,
        slope_mask,
        env['slope_threshold']
    )
    
    # Step 10: Calculate removed area
    removed_area_km2 = calculate_removed_area(removed_count, pixel_size_m=10)
    
    print(f'\n📏 False Positives Removed: {removed_area_km2:.2f} km²')
    
    # Step 11: Visualization
    visualize_before_after(
        confidence_before,
        confidence_after,
        removed_area_km2,
        output_path='output/task3_topographic_correction.png'
    )
    
    print('\n' + '=' * 70)
    print('✅ Task 3 Complete: Topographic Analysis — DEM & Slope Assessment')
    print('=' * 70)
    
    # Homework Discussion Placeholder
    print('\n' + '=' * 70)
    print('📋 HOMEWORK DISCUSSION')
    print('=' * 70)
    print('Q: Is the pre-disaster DEM still valid for a newly formed barrier lake?')
    print()
    print('A: [To be completed by student]')
    print('   Consider:')
    print('   - Copernicus DEM GLO-30 is based on 2011-2014 data')
    print('   - The 2025 barrier lake formed after a landslide')
    print('   - Landslides significantly alter terrain morphology')
    print('   - The dam structure and reservoir area did not exist in the pre-disaster DEM')
    print('   - Therefore, slope calculations may be inaccurate in the affected area')
    print('   - This is a known limitation of using pre-disaster DEMs for post-disaster analysis')
    print('   - Solutions: Post-disaster UAV LiDAR, satellite photogrammetry, or InSAR')
    print('=' * 70)


if __name__ == '__main__':
    main()
