# ARIA v5.0 Beta — 馬太鞍三幕審計器

> **ARIA**: Accessible Resilience & Intelligence Auditor — 可及韌性智慧稽核框架
> 
> 核心案例：2025 年花蓮縣光復鄉馬太鞍溪堰塞湖潰堤事件

---

## 一、專案簡介 (Introduction)

### 三幕式災害審計 (Three-Act Audit)

本專案利用歐洲太空總署 **Sentinel-2 L2A** 多光譜衛星影像（10–20 公尺解析度），對馬太鞍溪堰塞湖事件進行「時間–空間–光譜」三維度稽核：

| 幕次 (Act) | 時相 | 影像證據 | 物理意義 |
|:----------:|:----:|:---------|:---------|
| **Act 1 — Pre** | 2025-06-15 | 森林覆蓋、無水體 | 颱風侵襲前基線狀態 |
| **Act 2 — Mid** | 2025-09-11 | 堰塞湖形成 (~1.05 km²) | 崩塌體阻塞河道、高濁度水體 |
| **Act 3 — Post** | 2025-10-16 | 潰堤、土石流鋪面 (~2.34 km²) | 釋流後堆積物覆蓋光復鄉稻田 |

> **技術核心**：透過 **STAC API** 串流 **Microsoft Planetary Computer** 影像，使用 `stackstac` 建立 Lazy-loading 資料立方體 (Cube)，以最小化本機 I/O 負荷。

---

## 二、偵測物理邏輯 (Spectral Logic)

### 2.1 堰塞湖偵測 — 濁水體光譜特徵

```python
# S7: Barrier Lake Detection Logic
lake_mask = (
    (nir_pre > 0.25) &           # 事前為森林（高 NIR 反射）
    (nir_mid < 0.15) &           # 事後 NIR 驟降（水體吸收 0.8–1.0 μm）
    (blue_mid > 0.03) &          # 藍光散射（濁度指標）
    (green_mid > nir_mid) &      # 綠光 > NIR（排除雲陰干擾）
    upstream_gate                # 空間閘道：鎖定上游 121.33°E 以西
)
```

> **物理機制**：清水在近紅外 (NIR, B08) 幾乎完全吸收，反射率趨近於零；濁水因懸浮泥沙產生微弱後向散射，NIR ≈ 0.10–0.18，形成可與植被區分的閾值窗口。

### 2.2 山崩與土石流偵測 — 裸地與植被損失

| 災害類型 | 主波段指標 | 邏輯判斷 | 空間閘道 |
|---------|:----------:|:---------|:---------|
| **上游崩塌源** (Landslide) | NIR Drop + SWIR Surge | `nir_drop > 0.10` & `swir_post > 0.20` | 上游集水區 |
| **下游土石流** (Debris Flow) | NDVI↓ + BSI↑ | `ndvi_change > 0.25` & `bsi_change > 0.10` | 下游 121.35°E 以東 |

```python
# S8: Landslide Source (Bare Rock)
landslide_mask = (nir_pre - nir_post > 0.10) & (swir_post > 0.20)

# S9: Debris Flow (Wet Sediment over Paddy)
debris_mask = (
    (ndvi_pre - ndvi_post > 0.25) &      # 植被損失
    (bsi_post - bsi_pre > 0.10) &        # 裸土增加
    downstream_gate                       # 馬太鞍溪口下游
)
```

> **物理機制**：
> - **NIR Drop**：崩塌後裸露岩石/土壤之 NIR 反射率顯著低於森林冠層。
> - **SWIR Surge**：短波紅外 (SWIR, B11) 對水分敏感，濕潤崩積土產生高反射。
> - **BSI (Bare Soil Index)**：`(B11+B04)-(B08+B02)/(B11+B04)+(B08+B02)`，正值表裸土，負值表植被。

---

## 三、環境設定 (Installation & Setup)

### 3.1 相依套件

```bash
conda create -n gis-env python=3.11
conda activate gis-env
conda install -c conda-forge geopandas rasterio xarray rioxarray stackstac
pip install pystac-client planetary-computer google-generativeai python-dotenv
```

### 3.2 環境變數設定 (.env)

於專案根目錄建立 `.env` 檔案（**切勿提交至版本控制**）：

```bash
# API 金鑰
GEMINI_API_KEY=your_gemini_api_key_here

# 空間分析緩衝區（公尺）
BUFFER_SHELTER=100        # 避難所與崩塌源判斷距離
BUFFER_BOTTLENECK=200     # 瓶頸節點與崩塌源判斷距離
BUFFER_DEBRIS=0         # 土石流區嚴格相交判斷

# 三幕時間窗（ISO 8601）
PRE_EVENT=2025-06-15      # S2A_20250615 — 颱風前基線
MID_EVENT=2025-09-11      # S2C_20250911 — 堰塞湖峰值
POST_EVENT=2025-10-16     # S2B_20251016 — 潰堤後災情
```

### 3.3 專案檔案結構

```
gis_aria/
├── Week8-Student.ipynb          # 主分析筆記本（三幕稽核流程）
├── src/
│   └── guangfu_generator.py     # 光復鄉覆蓋層生成器
├── data/
│   ├── raw/                     # 原始輸入（避難所 CSV、河川 SHP）
│   ├── processed/               # 清理後資料（雨量 GeoJSON、避難所點位）
│   ├── scenarios/               # 情境分析資料（秀林鄉歷史案例）
│   ├── guangfu_overlay.gpkg     # W8 光復鄉 5 節點覆蓋圖
│   ├── guangfu_network.graphml  # 光復鄉 OSM 路網
│   └── xiulin_network.graphml   # W7 秀林鄉路網（歷史比較）
├── output/
│   ├── figures/                 # 視覺化成果（07–12 號圖表）
│   ├── vectors/                 # 向量成果（mataian_detections.gpkg）
│   ├── tables/                  # 數據表（impact_table.csv）
│   └── prompts/                 # AI 提示詞（ai_advisor_prompt.txt）
└── cache/                       # STAC API 查詢快取
```

---

## 四、空間稽核 (Spatial Audit)

### 4.1 多層次資產聯結 (Multi-Layer Asset Join)

將遙測萃取之災害遮罩與地面資產進行空間交集，計算「衝擊命中」矩陣：

```python
# S12: Spatial Join Logic
shelter_hit = gpd.sjoin(
    shelters_gdf,                  # W3 避難所（14 點，光復鄉）
    landslides_gdf.buffer(100),    # 崩塌源 100m 緩衝
    predicate='intersects'
)

bottleneck_hit = gpd.sjoin(
    top5_gdf,                      # W7 瓶頸（5 點，路網中介中心性 Top-5）
    landslides_gdf.buffer(200),   # 崩塌源 200m 緩衝
    predicate='intersects'
)

guangfu_hit = gpd.sjoin(
    guangfu_gdf,                   # W8 光復節點（5 必需點：車站、國小、鄉公所、橋樑、土石流區）
    debris_gdf,                    # 土石流遮罩（嚴格相交）
    predicate='intersects'
)
```

### 4.2 覆蓋缺口診斷 (Coverage Gap Analysis)

| 資產層級 | 樣本數 | 崩塌/土石流命中 | 命中率 |
|:--------:|:------:|:---------------:|:------:|
| **W3 避難所** | 14 | 4/14 | 28.6% |
| **W7 瓶頸點** | 5 | 5/5 | 100% |
| **W8 光復節點** | 5 | 4/5 | 80% |

> **政策意涵**：原 ARIA v4.0 僅覆蓋「花蓮市避難所 (W3)」與「秀林鄉路網瓶頸 (W7)」，未納入「光復鄉潛在災害走廊」。馬太鞍事件證明，災害發生於雷達覆蓋範圍之外，凸顯 W8 擴充之必要性。

### 4.3 輸出成果

| 檔案 | 說明 | 產出階段 |
|:-----|:-----|:--------:|
| `output/figures/12_coverage_gap_map.png` | 三幕災害疊加 W3/W7/W8 資產 | S12 |
| `output/tables/impact_table.csv` | 資產衝擊矩陣（Y/N 命中表） | S12 |
| `output/vectors/mataian_detections.gpkg` | 三圖層：堰塞湖、崩塌源、土石流 | S10 |
| `output/prompts/ai_advisor_prompt.txt` | 光譜情報官提示詞（供 LLM 產生作戰簡報） | S13 |

---

## 五、參考與致謝

- **衛星資料**：Copernicus Sentinel-2 (ESA) via Microsoft Planetary Computer
- **地形底圖**：OpenStreetMap (OSM)  via OSMnx
- **座標系統**：EPSG:32651 (UTM Zone 51N) 用於遙測分析；EPSG:3826 (TWD97/TM2) 用於國內圖資整合

---

*本專案為國立臺灣大學地理環境資源學系「地理資訊系統應用」課程成果，僅供學術研究使用。*
