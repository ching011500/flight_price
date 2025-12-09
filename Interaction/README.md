# Interaction Analysis Plan

本資料夾用來實驗「一維主效應 vs. 二維交互作用」對票價預測的影響。涵蓋四種艙等 × 三種模型：

- 短程：經濟艙、商務艙  
- 長程：經濟艙、商務艙  
- 模型：RandomForest (RF)、XGBoost (XGB)、Support Vector Regressor (SVR)

## 1. 資料來源與切分

- 來源檔：`merge_and_cleaned/final_data/short_flight.csv`、`long_flight.csv`
- 共同欄位：航段時段、停靠站、航空聯盟/公司、機型分類、假期、是否平日、飛行/停留/實際飛行時間、`competing_flights`、經濟/機場指標...
- 流程：先依 `航程 × 艙等` 切分資料，再輸入下游建模流程。確保所有欄位（含交互項）在四個子資料集中完全一致。

## 2. 特徵設計

### 2.1 一維主效應（Baseline）
1. 類別欄做 One-hot（`drop_first=True`，避免完全共線）。
2. 數值欄標準化（共用 `StandardScaler`）。
3. 保存欄位順序以便複製到四個艙等。

### 2.2 二維交互（Interaction）
1. 類別 × 類別：鎖定最具意義的組合，例如  
   - `出發時段 × 航空聯盟`  
   - `航段（出發/抵達時段） × 假期/是否為平日`  
   - `停靠站數量 × 航空聯盟`  
   - `航空公司組合 × 機型分類`
2. 類別 × 數值：以數值欄乘上類別 dummy（如 `午夜班機 × competing_flights`）或將類別分組後再乘。
3. 數值 × 數值：挑 `停留時間_分鐘 × competing_flights`、`實際飛行時間_分鐘 × 經濟指標` 等具備解釋性的項。
4. 控制欄位爆炸：限定組合清單、或在交互後以 `VarianceThreshold` / `SelectKBest` 剔除低資訊欄。

> 建議在合併資料（尚未依艙等拆分）階段創建交互欄，確保四個子集欄位對齊，再依艙等切開。

## 3. 建模流程

1. 共用函式產生 Baseline / Interaction 兩套特徵矩陣。
2. 固定隨機種子 & 70/30 train-test 切分（與既有 notebook 一致）。
3. RF / XGB：使用 `RandomizedSearchCV` 搜尋超參數（與舊設定接近，必要時增加 `max_features` 限制以避免交互特徵數太多）。
4. SVR：建議加入 `StandardScaler` + `Pipeline`；可針對 `C`、`gamma`、`epsilon` 做 log-uniform 搜尋。
5. 評估指標：`MSE`、`MAE`、`R²`。同時記錄訓練時間與參數。
6. 產出比較表：每個 `航程×艙等×模型` 會有 Baseline vs Interaction 的指標差值與勝出情況。

## 4. 現有檔案結構（精簡版）

```
Interaction/
  ├── README.md
  ├── scripts/
  │     ├── feature_builder.py     # 特徵生成 & 交互項建構
  │     └── run_interaction.py     # 批次跑 4 艙等 × 3 模型的 1D/2D 比較
  └── results/
        ├── metrics_raw.csv        # 各模型/特徵組合的完整指標
        ├── metrics_summary.csv    # 1D vs 2D 差異彙總 (rmse_delta/mae_delta/r2_gain)
        └── run_metadata.json      # 本次實驗元資料 (是否使用原生 xgboost 等)
```

> 註：目前未自動產生圖表，`plots/` 與 `notebooks/` 目錄已移除以保持精簡，如需 PDP/heatmap/SHAP 再加掛。

## 5. 二維交互評估建議（如需視覺化時參考）

- **Partial Dependence / ICE**：針對提升最多的交互組合畫 2D heatmap；與 1D PDP 比較。
- **SHAP interaction values** (針對 XGBoost)：可量化交互重要度。
- **統計檢定**：以同一測試集上的預測誤差，檢定 Baseline vs Interaction 是否顯著（例如配對 t-test）。

---

## 🚀 快速開始

1. 建議在虛擬環境中執行 `python -m Interaction.scripts.run_interaction`。  
2. 指令會自動針對四個艙等（短/長 × 經濟/商務）與三個模型（RF、SVR、XGB）完成 Baseline 與 Interaction 兩組訓練。  
3. 輸出結果：
   - `Interaction/results/metrics_raw.csv`：逐模型、逐特徵組合的完整指標與訓練統計。
   - `Interaction/results/metrics_summary.csv`：彙整表，包含 R²、RMSE、MAE 的一維 vs 二維差異。
   - `Interaction/results/run_metadata.json`：記錄此次實驗的基本資訊與是否使用原生 XGBoost。

> 若環境中無法安裝原生 `xgboost`（例如無網路），腳本會自動改用 `HistGradientBoostingRegressor` 作為備援，但仍沿用 `XGBoost` 命名，請於報告中註明。

之後可把上述 CSV/JSON 當成資料來源，製作互動視覺、報告表格或追加檢定。需要擴充 heatmap、SHAP 或整合 Streamlit，可再告訴我。***

