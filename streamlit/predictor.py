import streamlit as st
import pandas as pd
import random

# 設定頁面
st.set_page_config(page_title="國際機票價格預測系統", layout="centered")

# 頁面樣式
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
        background-color: #fdfdfd;
        color: #333333;
    }

    .stApp {
        max-width: 800px;
        margin: auto;
        padding-top: 2rem;
    }

    .stSelectbox, .stCheckbox, .stButton {
        background-color: #ffffff !important;
        border-radius: 12px;
        border: 1px solid #ddd !important;
        padding: 0.5rem;
    }

    .stButton > button {
        background-color: #f2f2f2;
        border: 1px solid #ccc;
        color: #333;
        border-radius: 12px;
        font-size: 16px;
        padding: 0.4rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# 標題
st.markdown("### 🛫 國際機票價格預測系統")

# 選單
departure = st.selectbox("請選擇出發機場", ["TPE（桃園）", "TSA（松山）"])
arrival = st.selectbox("請選擇抵達機場", [
    "NRT（成田）", "HND（羽田）", "SIN（新加坡）", "ICN（仁川）", "GMP（金浦）",
    "BKK（曼谷）", "HKG（香港）", "LHR（倫敦希斯洛）", "LAX（洛杉磯）",
    "FRA（法蘭克福）", "SYD（雪梨）", "CDG（巴黎戴高樂）", "ZRH（蘇黎世）", "JFK（紐約甘迺迪）"
])
stops = st.selectbox("請選擇轉機次數", [0, 1])
time_slot = st.selectbox("請選擇出發時段", ["凌晨", "早晨", "上午", "下午", "晚間"])
duration = st.checkbox("是否考慮飛行時間")

# 模擬按鈕
if st.button("📊 預測票價與建議"):

    flights = []
    for i in range(5):
        flight_no = f"CI{random.randint(100,999)}"
        airline = random.choice(["華航", "長榮", "星宇", "日航", "新加坡航空"])
        time_choice = random.choice(time_slot_range[time_slot])
        departure_time = f"2025/04/0{random.randint(1,9)} {time_choice}"
        price = random.randint(4000, 10000)
        predicted = random.randint(5000, 9000)
        lower = predicted - 800
        upper = predicted + 800
        suggestion = "✅ 推薦購買" if price < lower else ("⏳ 建議再等等" if price > upper else "🟡 價格合理")
        flights.append([flight_no, airline, departure_time, price, f"{predicted} ± 800", suggestion])

    df = pd.DataFrame(flights, columns=["航班編號", "航空公司", "出發時間", "實際票價", "預測區間", "建議"])

    # 顯示表格
    st.markdown("#### 🔍 符合條件的航班：")
    st.dataframe(df, use_container_width=True)

# 祝福語
st.markdown("✈️ <span style='font-size:18px;'>祝您旅途愉快，平安順利！</span>", unsafe_allow_html=True)
