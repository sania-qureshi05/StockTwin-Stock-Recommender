# StockTwin — Stock Similarity & Recommendation System

A Streamlit app that lets you select a listed stock and find similar stocks based on recent price returns, volatility, RSI, moving-average behaviour, and available sector/fundamental data. It provides similarity scores, explainable rule-based Buy / Hold / Avoid signals, a one-year price chart, and CSV export.

## Run on macOS

1. Download and extract `StockTwin_Stock_Recommender.zip`.
2. Open Terminal and enter the extracted folder. If it is in Downloads:

```bash
cd ~/Downloads/StockTwin_Stock_Recommender
```

3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Start the app:

```bash
python -m streamlit run app.py
```

Streamlit will show a local URL, usually http://localhost:8501.

## Notes

- The app retrieves public market data using Yahoo Finance through `yfinance`; an internet connection is required and some symbols may not return data.
- Indian stocks use Yahoo Finance NSE symbols ending in `.NS`; US symbols use their standard ticker.
- Similarity scores are relative to the stocks in the included universe, not a universal score.
- Buy / Hold / Avoid labels are simple educational rules, not personalized financial advice.
