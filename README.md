# ARIA v5.0: Matai'an Creek Impact Analysis & Diagnostic Log

## 1. Project Overview (專案概述)
本專案為 ARIA v5.0 的事後評估 (Post-Event Assessment)，旨在利用 Sentinel-2 多光譜衛星影像，重建花蓮馬太鞍溪上游堰塞湖的形成與潰決過程，並評估其對下游光復鄉關鍵基礎設施的衝擊。

## 2. Methodology & Thresholds (分析方法與閾值設定)
本分析利用 `stackstac` 處理雲端最佳化 GeoTIFF (COG)，並套用以下核心光譜閾值進行災情辨識：
* **堰塞湖 (Barrier Lake):** * `Pre-B08 > 0.25` (災前為植被)
    * `Mid-B08 < 0.18` & `Mid-B02 > 0.03` & `Mid-B03 > Mid-B08` (災中為高濁度水體)
* **崩塌源 (Landslide):** * `NIR Drop > 0.15` & `Post-SWIR > 0.25` & `Pre-B08 > 0.25` (植被消失轉為裸地)
    * *Minimum Area:* 2,000 m²
* **土石流 (Debris Flow):** * `NDVI Change > 0.25` & `BSI Change > 0.10` & `Pre-B08 > 0.20`
    * *Minimum Area:* 5,000 m²

## 3. Impact Assessment (衝擊檢核結果)
利用 `geopandas` 進行空間交集檢核 (Spatial Join)，並設定合理的緩衝區 (Buffer)：
* W3 避難所 (Shelters): 100m
* W7 交通瓶頸 (Bottlenecks): 200m
* 光復鄉節點 (Guangfu Overlay): 100m
* **崩塌地危害半徑 (Landslide Zone):** 200m

**🎯 光復鄉關鍵節點衝擊結果摘要：**
* **光復鄉公所 (Guangfu Township Office):** 被判定遭土石流 (`Debris Flow: Y`) 衝擊。
* **佛祖街沉積區中心 (Foxu Debris Zone):** 被判定遭土石流 (`Debris Flow: Y`) 衝擊。
* *備註：光復火車站與光復國小雖位於災區邊緣，但在本次嚴格的 100m 緩衝區與面積過濾條件下，被判定未直接遭受土石流覆蓋 (`Debris Flow: N`)。*

## 4. AI Diagnostic Log (除錯與系統極限分析)

在開發與執行過程中，遭遇了真實世界廣域遙測分析的典型挑戰，以下為關鍵診斷紀錄：

### A. 效能瓶頸與 Dask 死鎖 (The "Deadlock" Trap)
* **現象：** 在未限制 Bounding Box (BBOX) 或未採用單執行緒模式下，執行 Phase 3 的 `vectorize_and_filter` 函數時，系統出現嚴重的卡頓 (耗時超過 10 分鐘) 甚至死鎖。
* **診斷：** 這是因為處理縣市級的 10m 高解析度影像（多波段、多時期）時，資料量高達 1.5GB 以上。Python 的 `Dask` 延遲運算 (Lazy Evaluation) 在同時執行複雜幾何轉換與空間過濾時，引發了記憶體溢出或網路請求逾時 (SAS Token Expiration)。
* **解決方案：** 引入 `dask.diagnostics.ProgressBar` 監控真實下載進度，並將空間過濾邏輯（例如轉換至 EPSG:3826）延後至面積篩選（去除大量微小雜訊）之後執行，成功讓程式在合理時間內 (約 10 分鐘) 穩定跑完廣域運算。

### B. 坐標系轉換陷阱 (The CRS Mismatch)
* **現象：** 初始測試時，設定 `mid_cube.x < 121.35` 試圖過濾東西部，卻導致遮罩完全失效。
* **診斷：** 原始 Sentinel-2 影像位於 UTM 51N (EPSG:32651)，其 X 軸單位為公尺 (約在 280,000 左右)，而非經緯度。
* **解決方案：** 統一在 UTM 投影下進行面積計算與初步過濾，最後再轉換至 TWD97 (EPSG:3826) 進行 `x < 285000` (約等同 121.34E) 的物理邊界切割。

### C. 紅海現象與偽陽性探討 (The "Red Sea" Phenomenon)
*(請參考 `final_impact_map.png`)*
* **現象：** 最終衝擊表中，所有光復鄉節點的 `Landslide Hit` 均呈現 `Y`。視覺化地圖顯示，整個光復鄉平原幾乎被崩塌地的 200m 緩衝區（紅色區塊）淹沒。
* **診斷：** 這是純光譜檢核的極限。平原區秋季的農地收割或大面積裸露地，其光譜特徵 (NIR 下降、SWIR 升高) 與山坡地崩塌極為相似。當演算法未能嚴格將 `Landslide` 偵測限制在山區（例如漏掉平原區的空間 Mask），加上作業規定的 200m 巨大危害半徑，即產生了嚴重的偽陽性擴張。
* **優化建議：** 下一代 ARIA 模型必須整合 DEM (Digital Elevation Model)，過濾掉坡度小於一定閾值（例如 10 度）的區域，以徹底消除平原區的崩塌地誤判。

## 5. Conclusion (戰略建議)
本次 ARIA v5.0 分析證實，現有（v3, v7）集中於北花蓮的防災系統存在嚴重盲區。強烈建議重新分配資源，於馬太鞍溪流域等中南部高風險區增設預警節點。此外，光學衛星易受季風雲層干擾，建議未來導入 SAR (如 Sentinel-1) 提升全天候監測能力。