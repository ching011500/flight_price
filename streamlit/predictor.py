# streamlit_app.py
import os
import streamlit as st
import pandas as pd

# --------------------------
# 參數設定與映射
# --------------------------
departure_display = {
    "TPE（桃園）": "TPE",
    "TSA（松山）": "TSA"
}
arrival_mapping = {
    "NRT": "NRT（成田）", "HND": "HND（羽田）", "SIN": "SIN（新加坡）", "ICN": "ICN（仁川）",
    "GMP": "GMP（金浦）", "BKK": "BKK（曼谷）", "HKG": "HKG（香港）",
    "LHR": "LHR（倫敦希斯洛）", "LAX": "LAX（洛杉磯）", "FRA": "FRA（法蘭克福）",
    "SYD": "SYD（雪梨）", "CDG": "CDG（巴黎戴高樂）", "ZRH": "ZRH（蘇黎世）", "JFK": "JFK（紐約甘迺迪）"
}
alliance_map = {
    1: "星空聯盟", 2: "天合聯盟", 3: "寰宇一家", 4: "價值聯盟",
    5: "無聯盟傳統航空", 6: "無聯盟廉價航空"
}

# 出發時段映射：完整文字 → 短名
departure_time_display = {
    "凌晨班機（00:00–06:00）": "凌晨班機",
    "早晨班機（06:00–09:00）": "早晨班機",
    "上午班機（09:00–12:00）": "上午班機",
    "下午班機（12:00–18:00）": "下午班機",
    "晚間班機（18:00–00:00）": "晚間班機"
}

# --------------------------
# 長程／短程航班機場清單（判定模式用）
# --------------------------
short_airports = ["NRT","HND","SIN","ICN","GMP","BKK","HKG"]
long_airports  = ["LAX","JFK","LHR","CDG","FRA","SYD","ZRH"]

# --------------------------
# 資料路徑
# --------------------------
RAW_SHORT = 'cleaned_data/short_flight.csv'
RAW_LONG  = 'cleaned_data/long_flight.csv'
PRED_DIR  = 'predict/predict_data'

# --------------------------
# 快取函式
# --------------------------
@st.cache_data
def load_valid_combinations(raw_path):
    raw = pd.read_csv(raw_path)
    cols = ["出發機場代號","抵達機場代號","出發時段","抵達時段"]
    if '停靠站數量' in raw.columns:
        cols.append('停靠站數量')
    return raw[cols].drop_duplicates()

@st.cache_data
def load_all_predictions(mode='short'):
    subdir = 'short' if mode=='short' else 'long'
    dfs = []
    airports = short_airports if mode=='short' else long_airports
    for ap in airports:
        # 經濟艙
        e = pd.read_csv(os.path.join(PRED_DIR, subdir, f'eco_{ap}.csv'))
        e['艙等'] = '經濟艙'
        e['抵達機場代號'] = ap
        # 商務艙
        b = pd.read_csv(os.path.join(PRED_DIR, subdir, f'biz_{ap}.csv'))
        b['艙等'] = '商務艙'
        b['抵達機場代號'] = ap
        # 短程固定停靠站
        if mode=='short':
            e['停靠站數量'] = 0
            b['停靠站數量'] = 0
        dfs.extend([e, b])
    return pd.concat(dfs, ignore_index=True)

# --------------------------
# Streamlit 介面
# --------------------------
st.set_page_config(page_title="預測票價查詢系統", layout="centered")
st.title("✈️ 預測票價查詢系統")

# 1. 出發機場
dep_choice = st.selectbox("請選擇出發機場", list(departure_display.keys()), key="dep_airport")
departure = departure_display[dep_choice]

# 2. 抵達機場
arr_choice = st.selectbox("請選擇抵達機場", list(arrival_mapping.values()), key="arr_airport")
arrival = [k for k,v in arrival_mapping.items() if v==arr_choice][0]

# 3. 模式（長/短程）
mode = 'long' if arrival in long_airports else 'short'

# 4. 載入快取資料
valid = load_valid_combinations(RAW_LONG if mode=='long' else RAW_SHORT)
pred_all = load_all_predictions(mode)

# 5. 取當前航線 取當前航線
df = pred_all[
    (pred_all['出發機場代號']==departure) &
    (pred_all['抵達機場代號']==arrival)
]

# 6. 艙等
cabin_order = ['經濟艙','商務艙']
cabin_opts  = [c for c in cabin_order if c in df['艙等'].unique()]
cabin_opts += [c for c in df['艙等'].unique() if c not in cabin_opts]
cabin       = st.selectbox('艙等', cabin_opts, index=0, key="cabin_select")

# 7. 停靠站（長程）
if mode=='long':
    df_stop   = df[df['艙等']==cabin]
    stops_opts = sorted(df_stop['停靠站數量'].unique())
    stops     = st.selectbox('停靠站數量', stops_opts, key="stops_select")
else:
    stops = 0

# 8. 出發時段
df_dep  = df[(df['艙等']==cabin) & ((df['停靠站數量']==stops) if mode=='long' else True)]
avail   = df_dep['出發時段'].unique().tolist()
options = [full for full,short in departure_time_display.items() if short in avail]
others  = [t for t in avail if t not in departure_time_display.values()]
to_disp= options+others
dep_full = st.selectbox('出發時段', to_disp, index=0, key="dep_time_select")
dep_time = departure_time_display.get(dep_full, dep_full)

# 9. 聯盟
df_alm        = df[(df['艙等']==cabin)&(df['出發時段']==dep_time) & ((df['停靠站數量']==stops) if mode=='long' else True)]
alliance_vals = sorted(df_alm['航空聯盟'].unique())
alliance_disp = [alliance_map[v] for v in alliance_vals]
alm_choice    = st.selectbox('航空聯盟', alliance_disp, key="alliance_select")
rev_alm       = {v:k for k,v in alliance_map.items()}
alliance      = rev_alm[alm_choice]

# 10. 查詢
if st.button('🔍 查詢預測票價', key="search_btn"):
    res = df[
        (df['艙等']==cabin) &
        (df['出發時段']==dep_time) &
        (df['航空聯盟']==alliance)
    ]
    if mode=='long':
        res = res[res['停靠站數量']==stops]

    join_cols=['出發機場代號','抵達機場代號','出發時段','抵達時段']
    if mode=='long': join_cols.append('停靠站數量')
    res = res.merge(valid, on=join_cols, how='inner')

    if res.empty:
        st.warning('❌ 查無結果')
    else:
        res['預測_平均價格']=res['預測_平均價格'].round().astype(int)
        if mode=='long':
            disp_cols=['出發機場代號','抵達機場代號','出發時段','抵達時段','艙等','停靠站數量','航空聯盟','機型分類','假期','是否為平日','停留時間_分鐘','實際飛行時間_分鐘','competing_flights','預測_平均價格']
        else:
            disp_cols=['出發機場代號','抵達機場代號','出發時段','抵達時段','艙等','航空聯盟','機型分類','假期','是否為平日','飛行時間_分鐘','competing_flights','預測_平均價格']
        out=res[disp_cols].drop_duplicates()
        out['航空聯盟']=out['航空聯盟'].map(alliance_map)
        st.dataframe(out,use_container_width=True,hide_index=True)
