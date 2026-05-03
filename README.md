# ARIA v7.0 All-Weather Flood Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A multi-sensor flood detection system combining Sentinel-1 SAR and Sentinel-2 optical data for all-weather flood monitoring. Developed for Week 10 of the Remote Sensing Course, this system implements the ARIA v7.0 processing pipeline to detect and analyze flood events in cloudy conditions where optical-only methods fail.

## 🎯 Project Overview

This project demonstrates the critical value of SAR (Synthetic Aperture Radar) in typhoon disaster response by:

- **Penetrating cloud cover** to detect flooding when optical sensors are blocked
- **Fusing multi-sensor data** to create a 4-class confidence map for nuanced decision-making
- **Applying topographic correction** to remove radar shadow false positives on steep slopes
- **Generating strategic briefings** for emergency management and resource allocation

**Case Study:** 2025 Typhoon Fung-wong flood event in Matai'an Creek Basin, Hualien County, Taiwan

## ✨ Key Features

### Task 1: SAR All-Weather Flood Detection
- STAC API integration with Planetary Computer for Sentinel-1 RTC data streaming
- Linear to dB conversion with 10×log10 transformation
- 5×5 median filtering for speckle noise removal
- Morphological cleanup (opening + connected component filtering)
- Configurable SAR threshold (-14 dB default)

### Task 2: Sensor Fusion — Multi-Source Confidence Map
- Sentinel-2 L2A optical data with automatic cloud cover selection
- NDWI (Normalized Difference Water Index) computation
- SCL (Scene Classification Layer) cloud masking (values 3, 8, 9, 10)
- 4-class confidence matrix:
  - **Class 3:** High Confidence (SAR + Optical agree, clear sky)
  - **Class 2:** SAR Only (SAR detects water in cloudy areas)
  - **Class 1:** Optical Only (Optical detects water, SAR misses)
  - **Class 0:** No Detection

### Task 3: Topographic Analysis — DEM & Slope Assessment
- Copernicus DEM GLO-30 integration (30m → 10m resampling)
- Slope calculation using numpy gradient
- Topographic filtering (slope > 25°) to remove radar shadow artifacts
- Before/After comparison visualization

### Task 4: AI Strategic Briefing
- Emergency management advisor assessment
- Resource allocation recommendations
- Limitation analysis and additional data requirements
- ARIA v7.0 evolution report (W9 vs W10 comparison)

## 📁 Project Structure

```
week10/
├── Scripts/
│   ├── task1_sar_flood_detection.py    # SAR flood detection pipeline
│   ├── task2_sensor_fusion.py           # SAR+Optical fusion pipeline
│   └── task3_topographic_analysis.py    # DEM and slope analysis
├── output/
│   ├── task1_sar_results.png            # SAR processing results
│   ├── task2_confidence_map.png         # 4-class confidence map
│   └── task3_topographic_correction.png # Before/After comparison
├── Week10.ipynb                         # Main Jupyter notebook
├── .env                                 # Environment variables
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yang950104/gis-aria.git
   cd gis-aria
   ```

2. **Create virtual environment**
   ```bash
   python -m venv remo_w10
   source remo_w10/bin/activate  # On Windows: remo_w10\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install python-dotenv numpy matplotlib scipy pystac-client planetary-computer stackstac xarray
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   HUALIEN_BBOX=[121.27, 23.685, 121.31, 23.715]
   POST_DATE_RANGE=2025-09-10/2025-09-25
   SAR_THRESHOLD=-14
   NDWI_THRESHOLD=0.3
   SLOPE_THRESHOLD=25
   MIN_WATER_PIXELS=50
   ```

## 📖 Usage

### Option 1: Run Individual Scripts

```bash
# Task 1: SAR Flood Detection
python Scripts/task1_sar_flood_detection.py

# Task 2: Sensor Fusion
python Scripts/task2_sensor_fusion.py

# Task 3: Topographic Analysis
python Scripts/task3_topographic_analysis.py
```

### Option 2: Run in Jupyter Notebook

```bash
jupyter notebook Week10.ipynb
```

Then execute cells sequentially or use:
```python
%run Scripts/task1_sar_flood_detection.py
%run Scripts/task2_sensor_fusion.py
%run Scripts/task3_topographic_analysis.py
```

## 🔬 Technical Details

### Data Sources

| Sensor | Collection | Resolution | Revisit Time |
|--------|------------|------------|--------------|
| Sentinel-1 RTC | `sentinel-1-rtc` | 10m | 6 days |
| Sentinel-2 L2A | `sentinel-2-l2a` | 10m | 5 days |
| Copernicus DEM | `cop-dem-glo-30` | 30m (resampled to 10m) | Static |

### Processing Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Sentinel-1  │    │ Sentinel-2  │    │ Copernicus │
│   (SAR)     │    │  (Optical)  │    │    DEM      │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                  ↓                  ↓
  dB Conversion     NDWI + SCL        Slope Calc
     ↓                  ↓                  ↓
  Speckle Filter   Cloud Mask        Slope Mask
     ↓                  ↓                  ↓
  Water Mask      Water Mask         Topo Filter
     └────────┬─────────┘                  ↓
              ↓                    ┌────────┴────────┐
       4-Class Fusion            │  Filtered Map  │
              ↓                    └─────────────────┘
        Confidence Map
              ↓
     Strategic Briefing
```

### Key Algorithms

**SAR Processing:**
- Linear to dB: `dB = 10 * log10(linear_value)`
- Median filter: 5×5 kernel for speckle removal
- Threshold: VV < -14 dB for water detection
- Morphological cleanup: Binary opening + connected component filtering (min 50 pixels)

**Optical Processing:**
- NDWI: `(Green - NIR) / (Green + NIR)`
- Water threshold: NDWI > 0.3
- Cloud mask: SCL values 3, 8, 9, 10 (shadows and clouds)

**Topographic Processing:**
- Slope: `arctan(√[(dz/dx)² + (dz/dy)²])` in degrees
- Steep threshold: slope > 25°
- Filter: Remove water detections on steep slopes

## 📊 Results

### Detection Statistics (2025 Typhoon Fung-wong)

| Metric | Value |
|--------|-------|
| High Confidence Flood Area | 0.00 km² (clouds blocked optical) |
| SAR-Only (Cloudy) Flood Area | 1.15 km² |
| False Positives Removed | 0.17 km² (DEM limitation) |
| Cloud Cover | 13.5% (directly over disaster zone) |
| Total Detected Water | 1.17 km² |

### W9 vs W10 Comparison

| Aspect | Week 9 (Optical-Only) | Week 10 (Fused) |
|--------|----------------------|-----------------|
| Cloud Handling | Failed (0.00 km²) | Success (1.15 km²) |
| Confidence Classes | 2 | 4 |
| Topographic Correction | No | Yes |
| All-Weather Capability | No | Yes |

## ⚠️ Limitations

1. **DEM Temporal Mismatch**
   - Copernicus DEM based on 2011-2014 data
   - Does not account for 2025 landslide terrain changes
   - Topographic filter may remove true water from newly formed barrier lakes

2. **API Timeout Issues**
   - Planetary Computer STAC API may timeout during high load
   - Implemented retry mechanism with exponential backoff

3. **Single-Temporal Analysis**
   - Only post-event data analyzed
   - Cannot distinguish permanent water bodies from temporary flooding

4. **Resolution Constraints**
   - DEM (30m) resampled to 10m may not capture micro-topography
   - Slope calculations may be inaccurate in complex terrain

## 🤝 Contributing

This is a course project. For suggestions or improvements, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Planetary Computer** for providing STAC API and COG streaming
- **Copernicus Programme** for Sentinel-1, Sentinel-2, and DEM data
- **NASA JPL ARIA** for the original ARIA flood detection concept
- **Remote Sensing Course Week 10** instructors and teaching assistants

## 📚 References

- [Planetary Computer Documentation](https://planetarycomputer.microsoft.com/docs/)
- [Sentinel-1 RTC Product Specification](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar)
- [Sentinel-2 L2A Product Specification](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [Copernicus DEM Documentation](https://copernicus-dem-30m.readthedocs.io/)
- [ARIA Flood Maps](https://aria.jpl.nasa.gov/)

## 📧 Contact

For questions or inquiries, please open an issue on GitHub.

---

**Course:** Remote Sensing Week 10  
**Date:** 2025  
**Institution:** [Your Institution]
