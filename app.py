import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title='StockTwin | Stock Similarity', page_icon='📈', layout='wide')

st.markdown('''<style>
.stApp {background:#171615;color:#f5eee5}
[data-testid="stSidebar"] {background:#292019}
[data-testid="stMetric"] {background:#252321;padding:14px;border:1px solid #40382f;border-radius:12px}
div[data-testid="stDataFrame"] {border:1px solid #40382f;border-radius:10px}
.stButton>button {background:#e9c59a;color:#21180f;border:0;border-radius:9px;font-weight:700}
.small-note {color:#bdb3a7;font-size:0.9rem}
</style>''', unsafe_allow_html=True)

UNIVERSE = {
    'India (NSE)': {
        'Reliance Industries':'RELIANCE.NS','TCS':'TCS.NS','Infosys':'INFY.NS','HDFC Bank':'HDFCBANK.NS',
        'ICICI Bank':'ICICIBANK.NS','Bharti Airtel':'BHARTIARTL.NS','ITC':'ITC.NS','Larsen & Toubro':'LT.NS',
        'Wipro':'WIPRO.NS','HCL Technologies':'HCLTECH.NS','Tech Mahindra':'TECHM.NS','Tata Motors':'TATAMOTORS.NS',
        'Mahindra & Mahindra':'M&M.NS','Maruti Suzuki':'MARUTI.NS','Asian Paints':'ASIANPAINT.NS','Sun Pharma':'SUNPHARMA.NS',
        'Cipla':'CIPLA.NS','Tata Steel':'TATASTEEL.NS','Adani Ports':'ADANIPORTS.NS','Axis Bank':'AXISBANK.NS',
        'Kotak Mahindra Bank':'KOTAKBANK.NS','Bajaj Finance':'BAJFINANCE.NS','Titan':'TITAN.NS','UltraTech Cement':'ULTRACEMCO.NS',
        'NTPC':'NTPC.NS','Power Grid':'POWERGRID.NS','Zomato':'ETERNAL.NS','Coal India':'COALINDIA.NS',
        'Bajaj Auto':'BAJAJ-AUTO.NS','Nestle India':'NESTLEIND.NS'
    },
    'United States': {
        'Apple':'AAPL','Microsoft':'MSFT','NVIDIA':'NVDA','Amazon':'AMZN','Alphabet (Google)':'GOOGL',
        'Meta Platforms':'META','Tesla':'TSLA','Broadcom':'AVGO','JPMorgan Chase':'JPM','Walmart':'WMT',
        'Eli Lilly':'LLY','Visa':'V','Mastercard':'MA','Netflix':'NFLX','Coca-Cola':'KO','PepsiCo':'PEP',
        'Exxon Mobil':'XOM','Chevron':'CVX','Johnson & Johnson':'JNJ','Procter & Gamble':'PG',
        'Oracle':'ORCL','Salesforce':'CRM','AMD':'AMD','Intel':'INTC','Cisco':'CSCO','Adobe':'ADBE',
        'Bank of America':'BAC','Goldman Sachs':'GS','Costco':'COST','Uber':'UBER'
    }
}

@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, period='1y'):
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty or 'Close' not in df:
        return pd.DataFrame()
    return df.dropna(subset=['Close'])

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

def make_features(ticker, name):
    hist = get_history(ticker, '1y')
    info = get_info(ticker)
    if hist.empty or len(hist) < 30:
        return None
    close = hist['Close'].astype(float)
    ret = close.pct_change().dropna()
    latest = float(close.iloc[-1])
    def ret_n(n):
        if len(close) <= n: return np.nan
        return float(close.iloc[-1] / close.iloc[-n-1] - 1)
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    vol20 = ret.tail(20).std() * np.sqrt(252) if len(ret) >= 20 else np.nan
    # RSI, on the latest daily close series
    delta = close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100/(1+rs)).iloc[-1]) if len(close) >= 15 else np.nan
    features = {
        'Company': name, 'Ticker': ticker, 'Price': latest,
        '1M Return': ret_n(21), '3M Return': ret_n(63), '6M Return': ret_n(126), '1Y Return': ret_n(252),
        'Annual Volatility': float(vol20) if pd.notna(vol20) else np.nan,
        'RSI (14)': rsi, 'Price / MA20': float(latest/ma20-1) if pd.notna(ma20) and ma20 else np.nan,
        'Price / MA50': float(latest/ma50-1) if pd.notna(ma50) and ma50 else np.nan,
        'Sector': info.get('sector','Unknown'), 'Industry': info.get('industry','Unknown'),
        'Market Cap': info.get('marketCap', np.nan), 'PE Ratio': info.get('trailingPE', np.nan),
        'ROE': info.get('returnOnEquity', np.nan), 'Revenue Growth': info.get('revenueGrowth', np.nan),
        'History': hist
    }
    return features

def similarity_table(features):
    df = pd.DataFrame([{k:v for k,v in f.items() if k != 'History'} for f in features])
    if len(df) < 2: return pd.DataFrame()
    metric_cols = ['1M Return','3M Return','6M Return','1Y Return','Annual Volatility','RSI (14)','Price / MA20','Price / MA50','PE Ratio','ROE','Revenue Growth']
    matrix = df[metric_cols].apply(pd.to_numeric, errors='coerce')
    # robust scaling by z-score, with missing values imputed using the candidate median
    matrix = matrix.fillna(matrix.median()).fillna(0)
    std = matrix.std(ddof=0).replace(0, 1)
    z = ((matrix - matrix.mean()) / std).clip(-4,4)
    sectors = df['Sector'].fillna('Unknown').astype(str)
    industries = df['Industry'].fillna('Unknown').astype(str)
    rows=[]
    for i in range(1, len(df)):
        # Similarity blends standardized market/fundamental metrics and sector match.
        distance = float(np.sqrt(np.mean((z.iloc[0].values-z.iloc[i].values)**2)))
        metric_score = 100*np.exp(-distance/2.5)
        sector_bonus = 12 if sectors.iloc[0] != 'Unknown' and sectors.iloc[0] == sectors.iloc[i] else 0
        industry_bonus = 8 if industries.iloc[0] != 'Unknown' and industries.iloc[0] == industries.iloc[i] else 0
        score = min(99, metric_score + sector_bonus + industry_bonus)
        rec, why = recommendation(df.iloc[i])
        rows.append({'Company':df.iloc[i]['Company'],'Ticker':df.iloc[i]['Ticker'],'Sector':sectors.iloc[i],
                     'Similarity Score':round(score,1),'Price':df.iloc[i]['Price'],'1M Return':df.iloc[i]['1M Return'],
                     '3M Return':df.iloc[i]['3M Return'],'1Y Return':df.iloc[i]['1Y Return'],
                     'RSI (14)':df.iloc[i]['RSI (14)'],'Recommendation':rec,'Signal Explanation':why})
    return pd.DataFrame(rows).sort_values('Similarity Score',ascending=False)

def recommendation(row):
    r1 = row.get('1M Return', np.nan); r3=row.get('3M Return',np.nan); rsi=row.get('RSI (14)',np.nan)
    score=0; reasons=[]
    if pd.notna(r1):
        if r1 > 0: score += 1; reasons.append('positive 1-month return')
        else: score -= 1; reasons.append('negative 1-month return')
    if pd.notna(r3):
        if r3 > 0: score += 1; reasons.append('positive 3-month trend')
        else: score -= 1; reasons.append('negative 3-month trend')
    if pd.notna(rsi):
        if 40 <= rsi <= 65: score += 1; reasons.append('RSI in a moderate range')
        elif rsi > 75: score -= 1; reasons.append('RSI suggests overbought conditions')
        elif rsi < 30: reasons.append('RSI suggests oversold conditions; confirm before acting')
    label = 'Buy' if score >= 2 else ('Avoid' if score <= -2 else 'Hold / Watch')
    return label, '; '.join(reasons) if reasons else 'Insufficient indicators'

st.title('📈 StockTwin')
st.subheader('Stock Similarity & Recommendation System')
st.caption('Find stocks with similar market behaviour, compare key indicators, and review transparent Buy / Hold / Avoid signals.')

with st.sidebar:
    st.markdown('## ⚙️ Settings')
    market = st.selectbox('Market', list(UNIVERSE.keys()))
    st.markdown('### How it works')
    st.write('• Compares price returns and volatility\n• Includes RSI and moving-average behaviour\n• Uses available sector and fundamental data\n• Produces an explainable rule-based signal')
    st.markdown('---')
    st.caption('Data source: Yahoo Finance via yfinance. Availability and coverage can vary by ticker.')

companies = UNIVERSE[market]
opts = [f'{name} ({ticker})' for name,ticker in companies.items()]
selected = st.selectbox('Choose a stock to analyse', opts, index=opts.index('TCS (TCS.NS)') if market=='India (NSE)' else 0)
selected_name = selected.rsplit(' (',1)[0]
selected_ticker = selected.rsplit(' (',1)[1].rstrip(')')
max_candidates = st.slider('Number of similar stocks to show', min_value=5, max_value=20, value=10)

if st.button('Find Similar Stocks', type='primary'):
    st.session_state['run_analysis'] = True
    st.session_state['selected_name'] = selected_name
    st.session_state['selected_ticker'] = selected_ticker

if st.session_state.get('run_analysis'):
    base_name = st.session_state['selected_name']; base_ticker = st.session_state['selected_ticker']
    tickers = [(n,t) for n,t in companies.items() if t != base_ticker]
    all_to_fetch = [(base_name,base_ticker)] + tickers
    features=[]; failures=[]
    progress = st.progress(0, text='Downloading stock data…')
    for i,(name,ticker) in enumerate(all_to_fetch):
        try:
            f = make_features(ticker,name)
            if f: features.append(f)
            else: failures.append(ticker)
        except Exception:
            failures.append(ticker)
        progress.progress((i+1)/len(all_to_fetch), text=f'Loading {name}…')
    progress.empty()
    base = next((f for f in features if f['Ticker']==base_ticker), None)
    if base is None:
        st.error(f'Could not load enough recent price history for {base_name} ({base_ticker}). Check your internet connection or try another ticker.')
    else:
        table = similarity_table(features)
        info = get_info(base_ticker)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Latest Price', f"{base['Price']:,.2f}")
        c2.metric('1-Month Return', f"{base['1M Return']*100:+.2f}%" if pd.notna(base['1M Return']) else 'N/A')
        c3.metric('3-Month Return', f"{base['3M Return']*100:+.2f}%" if pd.notna(base['3M Return']) else 'N/A')
        c4.metric('Sector', base['Sector'] if base['Sector']!='Unknown' else 'Unavailable')
        st.markdown(f'### {base_name} ({base_ticker})')
        st.caption(f"Industry: {base['Industry']} · Market: {market}")
        hist=base['History'].reset_index()
        date_col = 'Date' if 'Date' in hist.columns else hist.columns[0]
        fig=px.line(hist,x=date_col,y='Close',title=f'{base_name} — adjusted closing price (1 year)')
        fig.update_layout(template='plotly_dark',paper_bgcolor='#171615',plot_bgcolor='#171615',font_color='#f5eee5',margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('### 🔎 Most Similar Stocks')
        if table.empty:
            st.warning('Not enough stocks returned valid data to calculate similarities.')
        else:
            shown=table.head(max_candidates).copy()
            display=shown[['Company','Ticker','Sector','Similarity Score','Price','1M Return','3M Return','1Y Return','RSI (14)','Recommendation']].copy()
            for col in ['1M Return','3M Return','1Y Return']:
                display[col]=display[col].map(lambda x: f'{x*100:+.2f}%' if pd.notna(x) else 'N/A')
            display['Price']=display['Price'].map(lambda x: f'{x:,.2f}' if pd.notna(x) else 'N/A')
            display['RSI (14)']=display['RSI (14)'].map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
            st.dataframe(display, use_container_width=True, hide_index=True)
            chart=px.bar(shown.sort_values('Similarity Score'),x='Similarity Score',y='Ticker',orientation='h',color='Similarity Score',
                         title='Similarity score (higher = more similar)', range_color=(0,100), color_continuous_scale=['#6c513e','#e9c59a'])
            chart.update_layout(template='plotly_dark',paper_bgcolor='#171615',plot_bgcolor='#171615',font_color='#f5eee5',margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(chart,use_container_width=True)
            st.markdown('### 🧭 Recommendation explanation')
            for _, row in shown.iterrows():
                with st.expander(f"{row['Ticker']} — {row['Recommendation']} · Similarity {row['Similarity Score']:.1f}/100"):
                    st.write(row['Signal Explanation'] + '.')
                    st.write(f"Sector: {row['Sector']}. Similarity is based on standardized returns, volatility, RSI, moving-average distance and available fundamentals, plus sector/industry matches when available.")
            csv=table.to_csv(index=False).encode('utf-8')
            st.download_button('⬇️ Download full results as CSV',csv,file_name=f'{base_ticker.replace(".","_")}_similar_stocks.csv',mime='text/csv')
        if failures:
            st.caption(f"Some tickers did not return data and were skipped ({len(failures)} skipped). This can happen due to data-source coverage or network limits.")

st.markdown('---')
st.markdown('**Important:** Similarity is not the same as investment quality. Buy / Hold / Avoid labels are educational, rule-based signals—not personalized financial advice or guaranteed predictions. Verify financial statements, valuation, risk, and current market information before investing.')
