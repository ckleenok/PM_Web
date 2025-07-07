# PySide6-based GUI app with MACD/Bollinger strategy using Yahooquery
import sys
import time
import pandas as pd
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                               QTableWidget, QTableWidgetItem, QLabel, QProgressBar)
from PySide6.QtCore import Qt, QTimer
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import streamlit as st
from supabase import create_client, Client

warnings.filterwarnings('ignore', category=FutureWarning, message="'S' is deprecated")
warnings.filterwarnings('ignore', category=FutureWarning, message=".*chained assignment.*")

# --- Strategy Parameters ---
MACD_BUY_RANGE = (-100, -80)
STOP_LOSS_THRESHOLD = -0.0175  # -1.75%
PROFIT_TAKE_THRESHOLD = 0.07  # +7%
MACD_SELL_THRESHOLD = 80

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.write("SUPABASE_URL:", repr(st.secrets["SUPABASE_URL"]))
st.write("SUPABASE_KEY:", repr(st.secrets["SUPABASE_KEY"]))

class StockAnalyzer(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portfolio Manager v3 6 month - 7JUL2025")
        self.setGeometry(100, 100, 1000, 700)

        main_layout = QVBoxLayout()

        # Top layout (20%)
        top_layout = QVBoxLayout()
        self.label = QLabel("Add Ticker:")
        top_layout.addWidget(self.label)

        self.ticker_input_row = QTableWidget()
        self.ticker_input_row.setColumnCount(1)
        self.ticker_input_row.setHorizontalHeaderLabels(["Enter Ticker"])
        self.ticker_input_row.setRowCount(1)
        self.ticker_input_row.setItem(0, 0, QTableWidgetItem(""))
        top_layout.addWidget(self.ticker_input_row)

        self.add_ticker_button = QPushButton("Add Ticker")
        self.add_ticker_button.clicked.connect(self.add_ticker_row)
        top_layout.addWidget(self.add_ticker_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_table)
        top_layout.addWidget(self.reset_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)

        # Bottom layout (80%)
        bottom_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(15)
        self.table.setHorizontalHeaderLabels(["Ticker", "Company Name", "Buy Date", "Buy Price", "Current Price", "Return", "", "", "", "", "", "Profit ≥ 7%", "Bollinger Touch", "Action", ""])
        self.table.setColumnWidth(1, 450)
        bottom_layout.addWidget(self.table)

        self.recalc_button = QPushButton("Re-Calculate")
        self.recalc_button.clicked.connect(self.recalculate_returns)
        bottom_layout.addWidget(self.recalc_button)

        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        bottom_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Saved")
        self.load_button.clicked.connect(self.load_results)
        bottom_layout.addWidget(self.load_button)

        main_layout.addLayout(top_layout, 1)
        main_layout.addLayout(bottom_layout, 4)
        self.setLayout(main_layout)

        self.progress_bar.setVisible(True)
        QTimer.singleShot(100, self.initial_load)

    def reset_table(self):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)

    def initial_load(self):
        self.load_results()
        self.refresh_existing_tickers()
        self.set_macd_headers_from_data()
        self.progress_bar.setVisible(False)

    def add_ticker_row(self):
        try:
            ticker_item = self.ticker_input_row.item(0, 0)
            ticker = ticker_item.text().strip().lower()
            if not ticker:
                print("No ticker entered.")
                return
            if ticker.isdigit() and len(ticker) == 6:
                ticker += ".ks"
            # rest of your existing logic to add ticker...
        except Exception as e:
            print(f"Error adding ticker {ticker}: {e}")

    def load_results(self):
        try:
            if os.path.exists("saved_results.csv"):
                df = pd.read_csv("saved_results.csv")
                expected_columns = ["Ticker", "Company Name", "Buy Date", "Buy Price", "Current Price", "Return", "MACD_0", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "Profit ≥ 7%", "Bollinger Touch", "Action"]
                if len(df.columns) >= len(expected_columns) and list(df.columns[:len(expected_columns)]) == expected_columns:
                    self.table.setRowCount(0)
                    for index, row in df.iterrows():
                        row_position = self.table.rowCount()
                        self.table.insertRow(row_position)
                        for i, val in enumerate(row):
                            self.table.setItem(row_position, i, QTableWidgetItem(str(val)))
                        delete_button = QPushButton("🗑️")
                        delete_button.clicked.connect(lambda _, row=row_position: self.table.removeRow(row))
                        self.table.setCellWidget(row_position, self.table.columnCount() - 1, delete_button)
                    print("Results loaded from saved_results.csv")
                else:
                    print("Warning: CSV format mismatch. Not loading.")
        except Exception as e:
            print(f"Failed to load results: {e}")

    def save_results(self):
        data = []
        for row in range(self.table.rowCount()):
            row_data = []
            for col in range(self.table.columnCount() - 1):
                item = self.table.item(row, col)
                row_data.append(item.text() if item else "")
            data.append(row_data)
        df = pd.DataFrame(data, columns=["Ticker", "Company Name", "Buy Date", "Buy Price", "Current Price", "Return", "MACD_0", "MACD_1", "MACD_2", "MACD_3", "MACD_4", "Profit ≥ 7%", "Bollinger Touch", "Action"])
        df.to_csv("saved_results.csv", index=False)
        print("Results saved to saved_results.csv")

    def recalculate_returns(self):
        for row in range(self.table.rowCount()):
            try:
                buy_price_item = self.table.item(row, 3)
                current_price_item = self.table.item(row, 4)
                if not buy_price_item or not current_price_item:
                    continue
                buy_price = float(buy_price_item.text())
                current_price = float(current_price_item.text())
                profit_pct = (current_price - buy_price) / buy_price
                self.table.setItem(row, 5, QTableWidgetItem(f"{profit_pct * 100:.2f}%"))

                profit_met = int(profit_pct >= PROFIT_TAKE_THRESHOLD)
                self.table.setItem(row, 11, QTableWidgetItem("✔️" if profit_met else "❌"))

                macd_values = [float(self.table.item(row, 6 + i).text()) if self.table.item(row, 6 + i) else None for i in range(5)]
                action = "🟢 HOLD"
                if all(val is not None for val in macd_values):
                    if macd_values[-1] >= MACD_SELL_THRESHOLD and macd_values[-1] < macd_values[-2]:
                        action = "✅ SELL"
                    elif MACD_BUY_RANGE[0] <= macd_values[-1] <= MACD_BUY_RANGE[1] and macd_values[-1] > macd_values[-2]:
                        action = "💰 BUY"
                self.table.setItem(row, 13, QTableWidgetItem(action))
            except Exception as e:
                print(f"Error recalculating row {row}: {e}")

    def reset_table(self):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)

    def set_macd_headers_from_data(self):
        today = datetime.now().date()
        macd_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)][::-1]
        headers = ["Ticker", "Company Name", "Buy Date", "Buy Price", "Current Price", "Return"] + macd_dates + ["Profit ≥ 7%", "Bollinger Touch", "Action", ""]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def refresh_existing_tickers(self):
        for row in range(self.table.rowCount()):
            ticker_item = self.table.item(row, 0)
            if ticker_item:
                ticker = ticker_item.text()
                self.update_ticker_data(ticker, row)

    def update_ticker_data(self, ticker, row_index):
        try:
            # 네이버 파이낸스용: ticker는 6자리 숫자(str)
            if ticker.endswith('.ks'):
                ticker_code = ticker.replace('.ks', '')
            else:
                ticker_code = ticker
            df = get_naver_price_history(ticker_code, months=6)
            print(f"[DEBUG] get_naver_price_history({ticker_code}) 결과:\n{df.head(10)}")
            company_name = get_naver_company_name(ticker_code)
            print(f"[DEBUG] get_naver_company_name({ticker_code}) 결과: {company_name}")
            if not df.empty and len(df) >= 26:
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

                macd_recent5 = df['MACD_Norm'].iloc[-5:].round(2).tolist()
                macd_dates = df.iloc[-5:].index.strftime('%Y-%m-%d').tolist()
                current_price = df['close'].iloc[-1]
                upper_band = df['Upper'].iloc[-1]
                bollinger_touch = "✔️" if current_price >= upper_band else "❌"

                self.table.setItem(row_index, 1, QTableWidgetItem(company_name))
                self.table.setItem(row_index, 4, QTableWidgetItem(f"{current_price:.2f}"))
                for i, macd_val in enumerate(macd_recent5):
                    self.table.setItem(row_index, 6 + i, QTableWidgetItem(str(macd_val)))

                self.table.setItem(row_index, 12, QTableWidgetItem(bollinger_touch))

                buy_price_item = self.table.item(row_index, 3)
                if buy_price_item and buy_price_item.text():
                    buy_price = float(buy_price_item.text())
                    profit_pct = (current_price - buy_price) / buy_price
                    self.table.setItem(row_index, 5, QTableWidgetItem(f"{profit_pct * 100:.2f}%"))
                    profit_met = int(profit_pct >= PROFIT_TAKE_THRESHOLD)
                    self.table.setItem(row_index, 11, QTableWidgetItem("✔️" if profit_met else "❌"))

                macd_values = [float(self.table.item(row_index, 6 + i).text()) if self.table.item(row_index, 6 + i) else None for i in range(5)]
                action = "🟢 HOLD"
                if all(val is not None for val in macd_values):
                    if macd_values[-1] >= MACD_SELL_THRESHOLD and macd_values[-1] < macd_values[-2]:
                        action = "✅ SELL"
                    elif MACD_BUY_RANGE[0] <= macd_values[-1] <= MACD_BUY_RANGE[1] and macd_values[-1] > macd_values[-2]:
                        action = "💰 BUY"
                self.table.setItem(row_index, 13, QTableWidgetItem(action))

        except Exception as e:
            print(f"Error updating data for {ticker}: {e}")

    def add_ticker_row(self):
        try:
            ticker_item = self.ticker_input_row.item(0, 0)
            ticker = ticker_item.text().strip()
            if not ticker:
                print("No ticker entered.")
                return
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            for i in range(self.table.columnCount()):
                self.table.setItem(row_position, i, QTableWidgetItem(""))
            self.table.setItem(row_position, 0, QTableWidgetItem(ticker))
            delete_button = QPushButton("🗑️")
            delete_button.clicked.connect(lambda _, row=row_position: self.table.removeRow(row))
            self.table.setCellWidget(row_position, self.table.columnCount() - 1, delete_button)
            self.update_ticker_data(ticker, row_position)
        except Exception as e:
            print(f"Error adding ticker {ticker}: {e}")

def get_naver_price_history(ticker, months=6):
    """
    네이버 파이낸스에서 6개월치 일별 종가와 날짜를 DataFrame으로 반환
    ticker: 종목코드(6자리, str)
    months: 가져올 개월 수
    """
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
        # 마지막 날짜가 start_date보다 이전이면 종료
        if (df['date'] < start_date).any():
            break
        page += 1
        time.sleep(0.2)  # 네이버 차단 방지
    if not dfs:
        return pd.DataFrame(columns=['date', 'close'])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df['date'] >= start_date]
    all_df = all_df.sort_values('date').reset_index(drop=True)
    all_df['close'] = all_df['close'].astype(str).str.replace(',', '').astype(float)
    return all_df

def get_naver_company_name(ticker):
    """
    네이버 파이낸스에서 종목코드로 회사명 반환
    """
    url = f"https://finance.naver.com/item/main.nhn?code={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    name_tag = soup.find('div', class_='wrap_company')
    if name_tag:
        name = name_tag.find('h2').text.strip().split('\n')[0]
        return name
    return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalyzer()
    window.show()
    sys.exit(app.exec())
