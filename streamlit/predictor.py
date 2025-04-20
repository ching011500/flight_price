# 根據上傳的資料建立完整 Streamlit 應用程式的程式碼
import streamlit as st
import pandas as pd

# 頁面設定
st.set_page_config(page_title="國際機票價格預測系統", layout="centered")

# 讀取資料
@st.cache_data
def load_data():
    return pd.read_csv("/Users/yuchingchen/Documents/專題/ci/ci_data/short_xgb_with_ci_str.csv")

df = load_data()

# 頁面樣式
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
        background-color: #fdfdfd;
        color: #333333;
    }
    .stApp {
        max-width: 900px;
        margin: auto;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛫 國際機票價格預測系統")

# 依序選單
# 出發機場選單（有中文）
departure_display = {
    "TPE（桃園）": "TPE",
    "TSA（松山）": "TSA"
}
departure_choice = st.selectbox("請選擇出發機場", list(departure_display.keys()))
departure = departure_display[departure_choice]
filtered_df = df[df["出發機場代號"] == departure]

# 抵達機場選單（根據出發地篩選，並顯示中文）
arrival_mapping = {
    "NRT": "NRT（成田）", "HND": "HND（羽田）", "SIN": "SIN（新加坡）", "ICN": "ICN（仁川）",
    "GMP": "GMP（金浦）", "BKK": "BKK（曼谷）", "HKG": "HKG（香港）", "LHR": "LHR（倫敦希斯洛）",
    "LAX": "LAX（洛杉磯）", "FRA": "FRA（法蘭克福）", "SYD": "SYD（雪梨）", "CDG": "CDG（巴黎戴高樂）",
    "ZRH": "ZRH（蘇黎世）", "JFK": "JFK（紐約甘迺迪）"
}
arrival_options = sorted(filtered_df["抵達機場代號"].dropna().unique())
arrival_display = {arrival_mapping[i]: i for i in arrival_options if i in arrival_mapping}
arrival_choice = st.selectbox("請選擇抵達機場", list(arrival_display.keys()))
arrival = arrival_display[arrival_choice]
filtered_df = filtered_df[filtered_df["抵達機場代號"] == arrival]

# 艙等
cabin_options = sorted(filtered_df["艙等"].dropna().unique())
cabin = st.selectbox("請選擇艙等", cabin_options)
filtered_df = filtered_df[filtered_df["艙等"] == cabin]

# 停靠站數量
stops_options = sorted(filtered_df["停靠站數量"].dropna().unique())
stops = st.selectbox("請選擇停靠站數量", stops_options)
filtered_df = filtered_df[filtered_df["停靠站數量"] == stops]

# 出發時段顯示選單對照
timeslot_display = {
    "凌晨班機（00:00–06:00）": "凌晨班機",
    "早晨班機（06:00–09:00）": "早晨班機",
    "上午班機（09:00–12:00）": "上午班機",
    "下午班機（12:00–18:00）": "下午班機",
    "晚間班機（18:00–00:00）": "晚間班機"
}

# 依前面條件篩選後取得可用出發時段
timeslot_options = sorted(filtered_df["出發時段"].dropna().unique())
timeslot_display_options = [k for k, v in timeslot_display.items() if v in timeslot_options]

# 出發時段選單
timeslot_choice = st.selectbox("請選擇出發時段", timeslot_display_options)
timeslot = timeslot_display[timeslot_choice]
filtered_df = filtered_df[filtered_df["出發時段"] == timeslot]

# 查詢按鈕
if st.button("🔍 查詢建議"):
    if filtered_df.empty:
        st.warning("找不到符合條件的航班。")
    else:
        result_df = filtered_df[[
            "出發時間", "抵達時間", "航班代碼", "航空公司",
            "實際價格", "預測值", "CI95下限", "CI95上限", "是否落在CI95"
        ]].copy()

        # 加上建議欄位
        def get_suggestion(row):
            if row["是否落在CI95"] == 1:
                return "✅ 推薦購買"
            else:
                return "⏳ 建議再等等"

        result_df["建議"] = result_df.apply(get_suggestion, axis=1)
        result_df["預測區間"] = result_df["CI95下限"].round(0).astype(int).astype(str) + " ~ " + result_df["CI95上限"].round(0).astype(int).astype(str)

        # 最終顯示欄位（新增抵達時間與航班代碼）
        result_df = result_df[[
            "出發時間", "抵達時間", "航班代碼", "航空公司",
            "實際價格", "預測區間", "建議"
        ]]

        st.markdown("### ✈️ 符合條件的航班建議如下：")
        st.dataframe(result_df, use_container_width=True)

# 祝福語
st.markdown("✈️ <span style='font-size:18px;'>祝您旅途愉快，平安順利！</span>", unsafe_allow_html=True)
