import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
from supabase import create_client, Client
import re
import uuid
from urllib.parse import unquote

# --- 컬럼 순서 및 상수 ---
COLUMNS = [
    "No.", "Company Name", "Buy Price", "Current Price", "Return",
    "MACD_4", "MACD_3", "MACD_2", "MACD_1", "MACD_0",
    "Profit ≥ 7%", "Bollinger Touch"
]

MACD_BUY_RANGE = (-100, -80)
STOP_LOSS_THRESHOLD = -0.0175
PROFIT_TAKE_THRESHOLD = 0.07
MACD_SELL_THRESHOLD = 80

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = "test_user"

def to_serializable(data):
    def convert_value(val):
        if isinstance(val, (np.generic, np.ndarray)):
            return val.item() if hasattr(val, 'item') else val.tolist()
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            return str(val)
        return val
    return [
        {k: convert_value(v) for k, v in row.items()}
        for row in data
    ]

def save_to_supabase(user_id, data):
    supabase.table("portfolio").insert({"user_id": user_id, "data": data}).execute()

def load_from_supabase(user_id):
    res = supabase.table("portfolio").select("data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    if res.data:
        df = pd.DataFrame(res.data[0]["data"])
        df = df.reindex(columns=[col for col in COLUMNS if col != "No."])
        df.insert(0, "No.", range(1, len(df) + 1))
        return df.to_dict('records')
    return []

def delete_supabase_data(user_id):
    supabase.table("portfolio").delete().eq("user_id", user_id).execute()

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

def is_safari():
    # Streamlit에서 User-Agent를 감지하여 사파리(특히 iOS) 여부를 반환
    user_agent = st.session_state.get('user_agent', None)
    if user_agent is None:
        # JS로 User-Agent를 세션에 저장하도록 유도 (최초 1회만)
        st.markdown("""
        <script>
        window.parent.postMessage({user_agent: navigator.userAgent}, '*');
        window.addEventListener('message', (event) => {
            if (event.data && event.data.user_agent) {
                window.location.search = '?user_agent=' + encodeURIComponent(event.data.user_agent);
            }
        });
        </script>
        """, unsafe_allow_html=True)
        return False
    # iPhone/iPad/Mac Safari 감지
    return bool(re.search(r"(iPhone|iPad|Macintosh).*Safari", user_agent))

def parse_number(val):
    if isinstance(val, str):
        val = val.replace(',', '')
    try:
        return float(val)
    except Exception:
        return None

def main():
    st.set_page_config(page_title="Portfolio Manager v3 (Web)", layout="wide")
    # 수동 새로고침, 선택 삭제, 수익률 재계산 버튼을 상단에 배치
    col_btns = st.columns([1,1,1])
    if col_btns[0].button("수동 새로고침"):
        st.rerun()
    delete_selected = col_btns[1].button("선택 삭제")
    recalc_return = col_btns[2].button("수익률 재계산")
    st.title("Portfolio Manager v3 (Web)")

    # 쿼리 파라미터에 user_agent가 있으면 세션에 저장하고, 한 번만 rerun
    query_user_agent = st.query_params.get('user_agent', [None])[0]
    if query_user_agent is not None and st.session_state.get('user_agent', None) is None:
        st.session_state['user_agent'] = unquote(query_user_agent)
        st.query_params.clear()
        st.rerun()

    if 'data' not in st.session_state:
        # 앱 첫 접속 시 Supabase에서 자동으로 데이터 불러오기
        data = load_from_supabase(USER_ID)
        if data:
            df = pd.DataFrame(data)
            df = df.reindex(columns=[col for col in COLUMNS if col != "No."])
            df.insert(0, "No.", range(1, len(df) + 1))
            st.session_state['data'] = df.drop(columns=["No."]).to_dict('records')
        else:
            st.session_state['data'] = []

    # 티커 입력
    with st.form(key="add_ticker_form"):
        cols = st.columns([1, 1, 1, 1, 1])
        ticker = cols[0].text_input("Add Ticker (6자리 숫자)", key="ticker_input")
        add_btn = cols[1].form_submit_button("Add Ticker")
        delete_all_btn = cols[2].form_submit_button("모든 데이터 삭제")
        save_btn = cols[3].form_submit_button("Supabase에 저장")
        load_btn = cols[4].form_submit_button("Supabase에서 불러오기")

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
                        # 볼린저 정규화: (close - Upper) 기준, 6개월간 min/max로 -100~100
                        bollinger_diff = df['close'] - df['Upper']
                        boll_min = bollinger_diff.min()
                        boll_max = bollinger_diff.max()
                        if boll_max - boll_min != 0:
                            boll_norm = (bollinger_diff.iloc[-1] - boll_min) / (boll_max - boll_min) * 200 - 100
                        else:
                            boll_norm = 0
                        current_price = df['close'].iloc[-1]
                        row_dict = {
                            "Company Name": company_name,
                            "Buy Price": "",
                            "Current Price": current_price,
                            "Return": "",
                            "MACD_4": macd_recent5[4] if len(macd_recent5) > 4 else "",
                            "MACD_3": macd_recent5[3] if len(macd_recent5) > 3 else "",
                            "MACD_2": macd_recent5[2] if len(macd_recent5) > 2 else "",
                            "MACD_1": macd_recent5[1] if len(macd_recent5) > 1 else "",
                            "MACD_0": macd_recent5[0] if len(macd_recent5) > 0 else "",
                            "Profit ≥ 7%": "",
                            "Bollinger Touch": round(boll_norm, 2),
                        }
                        ordered_row = {col: row_dict.get(col, "") for col in COLUMNS if col != "No."}
                        st.session_state['data'].append(ordered_row)
                    else:
                        st.warning("데이터를 불러올 수 없습니다.")
            else:
                st.warning("6자리 숫자 티커를 입력하세요.")

    if delete_all_btn:
        delete_supabase_data(USER_ID)
        st.session_state['data'] = []
        st.success("모든 데이터가 완전히 삭제되었습니다.")
    if save_btn:
        save_to_supabase(USER_ID, to_serializable(st.session_state['data']))
        st.success("Supabase에 저장 완료!")
    if load_btn:
        data = load_from_supabase(USER_ID)
        if data:
            df = pd.DataFrame(data)
            df = df.reindex(columns=[col for col in COLUMNS if col != "No."])
            df.insert(0, "No.", range(1, len(df) + 1))
            st.session_state['data'] = df.drop(columns=["No."]).to_dict('records')
            st.success("Supabase에서 불러오기 완료!")
        else:
            st.warning("데이터를 불러올 수 없습니다.")

    # 표 표시 및 편집
    if st.session_state['data']:
        # 최신 가격/수익률/Profit ≥ 7%를 실시간으로 계산해서 표시
        display_rows = []
        today = datetime.now().date()
        macd_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(4, -1, -1)]
        macd_col_map = {f"MACD_{i}": macd_dates[i] for i in range(5)}
        total_tickers = len(st.session_state['data'])
        status_placeholder = st.empty()
        for idx, row in enumerate(st.session_state['data']):
            ticker_code = None
            # 티커코드 추출 (Company Name에서 추출 불가시, row에 별도 저장 필요)
            # 여기서는 Company Name이 아닌, row에 'Ticker' 필드가 있다고 가정
            if 'Ticker' in row:
                ticker_code = row['Ticker']
            elif 'Company Name' in row:
                # Company Name만 있을 경우, 티커코드 추출 불가(추가 구현 필요)
                ticker_code = None
            buy_price = parse_number(row.get('Buy Price', ''))
            company_name = row.get('Company Name', '')
            status_placeholder.info(f"처리 중: {company_name} ({ticker_code if ticker_code else ''}) [{idx+1}/{total_tickers}]")
            time.sleep(0.1)
            # 네이버에서 최신 가격 및 지표 받아오기
            current_price = ''
            profit_pct = ''
            profit_flag = ''
            macd_recent5 = [None]*5
            boll_norm = ''
            if ticker_code:
                df = get_naver_price_history(ticker_code, months=6)
                if not df.empty:
                    df = calculate_indicators(df)
                    current_price = df['close'].iloc[-1]
                    if buy_price:
                        profit_pct = (current_price - buy_price) / buy_price * 100
                        profit_flag = "✔️" if profit_pct >= 7 else "❌"
                    else:
                        profit_pct = ''
                        profit_flag = ''
                    macd_recent5 = df['MACD_Norm'].iloc[-5:].round(2).tolist()
                    bollinger_diff = df['close'] - df['Upper']
                    boll_min = bollinger_diff.min()
                    boll_max = bollinger_diff.max()
                    if boll_max - boll_min != 0:
                        boll_norm = (bollinger_diff.iloc[-1] - boll_min) / (boll_max - boll_min) * 200 - 100
                    else:
                        boll_norm = 0
            display_row = {
                "No.": "",
                "Company Name": company_name,
                "Buy Price": f"{int(buy_price):,}" if buy_price else '',
                "Current Price": f"{int(current_price):,}" if current_price else '',
                "Return": f"{profit_pct:.2f}%" if profit_pct != '' else '',
                macd_col_map["MACD_4"]: macd_recent5[4] if macd_recent5[4] is not None else '',
                macd_col_map["MACD_3"]: macd_recent5[3] if macd_recent5[3] is not None else '',
                macd_col_map["MACD_2"]: macd_recent5[2] if macd_recent5[2] is not None else '',
                macd_col_map["MACD_1"]: macd_recent5[1] if macd_recent5[1] is not None else '',
                macd_col_map["MACD_0"]: macd_recent5[0] if macd_recent5[0] is not None else '',
                "Profit ≥ 7%": profit_flag,
                "Bollinger Touch": round(boll_norm, 2) if boll_norm != '' else '',
            }
            display_rows.append(display_row)
        status_placeholder.empty()
        display_columns = [
            "No.", "Company Name", "Buy Price", "Current Price", "Return",
            macd_dates[0], macd_dates[1], macd_dates[2], macd_dates[3], macd_dates[4],
            "Profit ≥ 7%", "Bollinger Touch"
        ]
        df_display = pd.DataFrame(display_rows)
        df_display = df_display.reindex(columns=[col for col in display_columns if col != "No."])
        df_display.insert(0, "No.", range(1, len(df_display) + 1))
        # 체크박스 컬럼 추가
        if 'selected' not in df_display.columns:
            df_display.insert(0, 'selected', False)
        else:
            df_display['selected'] = df_display['selected'].fillna(False)

        if is_safari():
            st.info('iOS 사파리에서는 표가 읽기 전용으로 표시됩니다.')
            st.dataframe(df_display, height=1000)
        else:
            edited_df = st.data_editor(df_display, num_rows="dynamic", key="data_editor", height=1000)
            # 선택 삭제 버튼
            if delete_selected:
                filtered_df = edited_df[~edited_df['selected']].reset_index(drop=True)
                reverse_macd_col_map = {v: k for k, v in macd_col_map.items()}
                filtered_df = filtered_df.rename(columns=reverse_macd_col_map)
                st.session_state['data'] = filtered_df.drop(columns=["No.", "selected"]).to_dict('records')
                save_to_supabase(USER_ID, to_serializable(st.session_state['data']))
                st.rerun()
            if recalc_return:
                for i, row in edited_df.iterrows():
                    try:
                        buy_price = parse_number(row["Buy Price"])
                        current_price = parse_number(row["Current Price"])
                        if buy_price:
                            profit_pct = (current_price - buy_price) / buy_price * 100
                            edited_df.at[i, "Return"] = f"{profit_pct:.2f}%"
                            edited_df.at[i, "Profit ≥ 7%"] = "✔️" if profit_pct >= 7 else "❌"
                        else:
                            edited_df.at[i, "Return"] = ""
                            edited_df.at[i, "Profit ≥ 7%"] = ""
                    except Exception:
                        edited_df.at[i, "Return"] = ""
                        edited_df.at[i, "Profit ≥ 7%"] = ""
                reverse_macd_col_map = {v: k for k, v in macd_col_map.items()}
                edited_df = edited_df.rename(columns=reverse_macd_col_map)
                st.session_state['data'] = edited_df.drop(columns=["No.", "selected"]).to_dict('records')
                st.rerun()

if __name__ == "__main__":
    main() 