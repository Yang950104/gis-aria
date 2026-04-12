# ARIA 避難所容量分析專案

> **ARIA**: Accessibility, Readiness, Impact, Availability - 避難所可及性與風險評估框架

## 專案概述

本專案針對花蓮縣、宜蘭縣避難所進行容量與雨量風險分析，整合氣象觀測資料與避難所點位，建立空間預測模型。

## 資料結構

```
.
├── aria_v4.ipynb          # 主分析 notebook
├── data/
│   ├── processed/         # 處理後資料
│   │   ├── gaemi_rainfall.geojson      # 凱米颱風雨量站
│   │   ├── plumrain_rainfall.geojson   # 梅雨雨量站
│   │   ├── gaemi_kriging_rainfall.tif  # Kriging 雨量內插
│   │   └── cleaned_shelter_data.csv     # 清理後避難所資料
│   ├── raw/               # 原始資料
│   │   ├── 避難收容處所點位檔案v9.csv   # 避難所點位
│   │   └── RIVREGLN.*     # 河川流域 shapefile
│   └── scenarios/         # 情境模擬資料 (待建立)
├── output/
│   └── geotiff/           # 輸出柵格資料
│       ├── gaemi_kriging_rainfall.tif
│       ├── gaemi_kriging_variance.tif
│       └── gaemi_rf_rainfall.tif
└── cache/                 # 快取資料
```

## 資料來源

| 資料類型 | 來源 | 格式 |
|---------|------|------|
| 雨量觀測 | 凱米颱風(2024-07-25)、梅雨事件 | GeoJSON |
| 避難所 | 內政部避難收容處所點位檔案 v9 | CSV |
| 河川流域 | 水利署流域邊界 | Shapefile |

## 技術棧

- **Python**: geopandas, rasterio, scipy, sklearn
- **空間分析**: Kriging 內插、隨機森林預測
- **座標系統**: EPSG:3826 (台灣二度分帶 TWD97/TM2)

## 分析流程

1. **資料預處理**: 清理避難所資料，轉換座標系統
2. **雨量內插**: 使用 Kriging 與隨機森林進行空間預測
3. **風險評估**: 整合避難所容量與降雨風險
4. **視覺化輸出**: 產製 GeoTIFF 與分析圖表

## 使用方式

```bash
# 啟動虛擬環境
conda activate gis-env

# 開啟 notebook
jupyter lab aria_v4.ipynb
```

## 輸出成果

- `gaemi_kriging_rainfall.tif` - Kriging 雨量預測表面
- `gaemi_rf_rainfall.tif` - 隨機森林雨量預測表面
- `xiulin_bottlenecks.png` - 秀林鄉瓶頸分析圖

## 開發者

GIS 應用課程 Week 6 - Prediction Shootout

---

*本專案資料僅供學術研究使用*
