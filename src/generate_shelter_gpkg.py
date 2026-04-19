# generate_shelter_gpkg.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# 1. 設定檔案路徑
# 依據截圖，您的檔案名為 cleaned_shelter_data.csv，請確認其位於 data/processed/ 或其他確切位置
csv_path = "data/processed/cleaned_shelter_data.csv" 
# 若檔案位於 raw 資料夾，請改為：
# csv_path = "data/raw/避難收容處所點位檔案v9 (1).csv"

try:
    # 讀取 CSV (政府開放資料常使用 utf-8-sig 或 big5)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"找不到檔案：{csv_path}。請確認路徑是否正確。")
    exit()
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='big5')

# 2. 尋找目標欄位名稱 (容錯處理，避免欄位名稱微小差異)
county_col = [col for col in df.columns if '縣市' in col][0]
name_col = [col for col in df.columns if '名稱' in col or '處所' in col][0]
lon_col = [col for col in df.columns if '經度' in col][0]
lat_col = [col for col in df.columns if '緯度' in col][0]

# 3. 篩選花蓮縣市資料
df_hualien = df[df[county_col].str.contains('花蓮', na=False)].copy()

# 4. 清理並轉換經緯度格式
df_hualien = df_hualien.dropna(subset=[lon_col, lat_col])
df_hualien[lon_col] = pd.to_numeric(df_hualien[lon_col], errors='coerce')
df_hualien[lat_col] = pd.to_numeric(df_hualien[lat_col], errors='coerce')
df_hualien = df_hualien.dropna(subset=[lon_col, lat_col])

# 5. 建立後續分析必需的標準欄位
df_hualien['name'] = "Shelter_" + df_hualien.index.astype(str) # 建立唯一英文 ID
df_hualien['cn_name'] = df_hualien[name_col]
df_hualien['node_type'] = 'shelter'

# 6. 轉換為 GeoDataFrame (初始座標系統設定為 WGS84 / EPSG:4326)
geometry = [Point(xy) for xy in zip(df_hualien[lon_col], df_hualien[lat_col])]
gdf = gpd.GeoDataFrame(df_hualien, geometry=geometry, crs="EPSG:4326")

# 7. 轉換座標系統為台灣二度分帶 (TWD97 / EPSG:3826)
gdf_3826 = gdf.to_crs("EPSG:3826")

# 8. 輸出為 GeoPackage
output_path = Path("data/shelters_hualien.gpkg")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 僅保留所需欄位以減小檔案體積並避免欄位編碼問題
columns_to_keep = ['name', 'cn_name', 'node_type', 'geometry']
gdf_3826[columns_to_keep].to_file(output_path, driver="GPKG")

print(f"✅ 檔案建立成功：{output_path}")
print(f"📊 共處理 {len(gdf_3826)} 筆花蓮地區避難所資料。")