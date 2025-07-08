import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os

# --- Supabase 연동 ---
# pip install supabase
from supabase import create_client, Client

# 아래 두 값을 본인 Supabase 콘솔에서 복사해 입력하세요
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_ID = "test_user"  # 로그인 기능 없으므로 임시 고정

def save_to_supabase(user_id, data):
    supabase.table("portfolio").insert({"user_id": user_id, "data": data}).execute()

def load_from_supabase(user_id):
    res = supabase.table("portfolio").select("data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    if res.data:
        return res.data[0]["data"]
    return []

# --- 크롤링 함수 (main.py에서 복사) ---
def get_naver_price_history(ticker, months=6):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    dfs = []
    page = 1
    while True:
        url = f"https://finance.naver.com/item/sise_day.nhn?code={ticker}&page={page}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', class_='type2')
        if table is None:
            break
        df = pd.read_html(str(table))[0]
        df = df.dropna()
        if df.empty:
            break
        df = df.rename(columns={'날짜': 'date', '종가': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'close']]
        dfs.append(df)
        if (df['date'] < start_date).any():
            break
        page += 1
        time.sleep(0.2)
    if not dfs:
        return pd.DataFrame(columns=['date', 'close'])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df['date'] >= start_date]
    all_df = all_df.sort_values('date').reset_index(drop=True)
    all_df['close'] = all_df['close'].astype(str).str.replace(',', '').astype(float)
    return all_df

def get_naver_company_name(ticker):
    url = f"https://finance.naver.com/item/main.nhn?code={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    name_tag = soup.find('div', class_='wrap_company')
    if name_tag:
        name = name_tag.find('h2').text.strip().split('\n')[0]
        return name
    return ""

# --- 전략 파라미터 ---
MACD_BUY_RANGE = (-100, -80)
STOP_LOSS_THRESHOLD = -0.0175
PROFIT_TAKE_THRESHOLD = 0.07
MACD_SELL_THRESHOLD = 80

# --- 지표 계산 함수 ---
def calculate_indicators(df):
    df = df.copy()
    df = df.set_index('date')
    df.sort_index(inplace=True)
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Gap'] = df['MACD'] - df['Signal']
    macd_range = df['MACD_Gap'].max() - df['MACD_Gap'].min()
    df['MACD_Norm'] = (df['MACD_Gap'] - df['MACD_Gap'].min()) / macd_range * 200 - 100 if macd_range != 0 else 0
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['STD20'] = df['close'].rolling(window=20, min_periods=1).std()
    df['Upper'] = df['MA20'] + 2 * df['STD20']
    return df

# --- Streamlit 앱 ---
def main():
    st.set_page_config(page_title="Portfolio Manager v3 (Web)", layout="wide")
    st.title("Portfolio Manager v3 (Web)")

    if 'data' not in st.session_state:
        st.session_state['data'] = []

    # 티커 입력
    with st.form(key="add_ticker_form"):
        cols = st.columns([2,1])
        ticker = cols[0].text_input("Add Ticker (6자리 숫자)")
        add_btn = cols[1].form_submit_button("Add Ticker")

    if add_btn:
        if ticker:
            if ticker.isdigit() and len(ticker) == 6:
                ticker_code = ticker
                with st.spinner("데이터 불러오는 중..."):
                    df = get_naver_price_history(ticker_code, months=6)
                    if not df.empty:
                        company_name = get_naver_company_name(ticker_code)
                        df = calculate_indicators(df)
                        macd_recent5 = df['MACD_Norm'].iloc[-5:].round(2).tolist()
                        macd_dates = df.iloc[-5:].index.strftime('%Y-%m-%d').tolist()
                        current_price = df['close'].iloc[-1]
                        upper_band = df['Upper'].iloc[-1]
                        bollinger_touch = "✔️" if current_price >= upper_band else "❌"
                        st.session_state['data'].append({
                            "Ticker": ticker_code,
                            "Company Name": company_name,
                            "Buy Date": "",
                            "Buy Price": "",
                            "Current Price": current_price,
                            "Return": "",
                            **{f"MACD_{i}": v for i, v in enumerate(macd_recent5)},
                            "Profit ≥ 7%": "",
                            "Bollinger Touch": bollinger_touch,
                            "Action": ""
                        })
                    else:
                        st.warning("데이터를 불러올 수 없습니다.")
            else:
                st.warning("6자리 숫자 티커를 입력하세요.")

    # 표 표시 및 편집
    if st.session_state['data']:
        df = pd.DataFrame(st.session_state['data'])
        # Buy Date, Buy Price는 직접 입력 가능하게 (columns 인자 제거)
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            key="data_editor"
        )

        # 수익률, Profit ≥ 7%, Action 자동 계산
        for i, row in edited_df.iterrows():
            try:
                buy_price = float(row["Buy Price"]) if row["Buy Price"] != "" else None
                current_price = float(row["Current Price"])
                if buy_price:
                    profit_pct = (current_price - buy_price) / buy_price
                    edited_df.at[i, "Return"] = f"{profit_pct * 100:.2f}%"
                    profit_met = int(profit_pct >= PROFIT_TAKE_THRESHOLD)
                    edited_df.at[i, "Profit ≥ 7%"] = "✔️" if profit_met else "❌"
                else:
                    edited_df.at[i, "Return"] = ""
                    edited_df.at[i, "Profit ≥ 7%"] = ""
                # Action 계산
                macd_values = [float(row.get(f"MACD_{j}", 0)) for j in range(5)]
                action = "🟢 HOLD"
                if all(isinstance(val, float) for val in macd_values):
                    if macd_values[-1] >= MACD_SELL_THRESHOLD and macd_values[-1] < macd_values[-2]:
                        action = "✅ SELL"
                    elif MACD_BUY_RANGE[0] <= macd_values[-1] <= MACD_BUY_RANGE[1] and macd_values[-1] > macd_values[-2]:
                        action = "💰 BUY"
                edited_df.at[i, "Action"] = action
            except Exception as e:
                edited_df.at[i, "Return"] = ""
                edited_df.at[i, "Profit ≥ 7%"] = ""
                edited_df.at[i, "Action"] = ""
        # 표 다시 표시
        st.dataframe(edited_df, use_container_width=True)
        # 세션 상태 업데이트
        st.session_state['data'] = edited_df.to_dict('records')

    # 저장/불러오기/리셋/Supabase 버튼
    cols2 = st.columns(3)
    if cols2[0].button("Reset"):
        st.session_state['data'] = []
        st.success("초기화 완료!")
    if cols2[1].button("Supabase에 저장"):
        save_to_supabase(USER_ID, st.session_state['data'])
        st.success("Supabase에 저장 완료!")
    if cols2[2].button("Supabase에서 불러오기"):
        st.session_state['data'] = load_from_supabase(USER_ID)
        st.success("Supabase에서 불러오기 완료!")

if __name__ == "__main__":
    main() 