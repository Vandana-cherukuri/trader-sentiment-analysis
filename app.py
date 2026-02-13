import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")
st.title(" Trader Performance vs Market Sentiment")


sentiment_path = "data/fear_greed.csv"
trades_path = "data/trades.csv"

if not Path(sentiment_path).exists() or not Path(trades_path).exists():
    st.error("⚠️ CSV files missing in the 'data/' folder.")
    st.stop()

sentiment = pd.read_csv(sentiment_path)
trades = pd.read_csv(trades_path)

def detect_column(df, possible_names, col_type):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

account_col = detect_column(trades, ['account','Account','account_id','user','User'], "account")
if account_col is None:
    st.error(f"⚠️ Could not find account column. Columns in trades: {', '.join(trades.columns)}")
    st.stop()


size_col = detect_column(trades, ['size','Size','Trade Size','Size Tokens','Size USD'], "trade size")
if size_col is None:
    st.error(f"⚠️ Could not find trade size column. Columns in trades: {', '.join(trades.columns)}")
    st.stop()

leverage_col = detect_column(trades, ['leverage','Leverage'], "leverage")
if leverage_col is None:
    trades['leverage'] = 1
    leverage_col = 'leverage'

st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
col1.metric("Sentiment Rows", sentiment.shape[0])
col2.metric("Trades Rows", trades.shape[0])

st.write("### Missing Values")
st.write("Sentiment:", sentiment.isnull().sum())
st.write("Trades:", trades.isnull().sum())

st.write("### Duplicate Rows")
st.write("Sentiment:", sentiment.duplicated().sum())
st.write("Trades:", trades.duplicated().sum())


sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce')
trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True, errors='coerce')

trades['date'] = trades['Timestamp IST'].dt.date
sentiment['date'] = sentiment['date'].dt.date
df = pd.merge(trades, sentiment, on='date', how='inner')


st.subheader("Filter by Date")
min_date = df['date'].min()
max_date = df['date'].max()
start_date, end_date = st.date_input("Select date range", [min_date, max_date])
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]


df['win'] = df['Closed PnL'] > 0

daily_metrics = df.groupby([account_col,'date']).agg(
    daily_PnL=('Closed PnL','sum'),
    win_rate=('win','mean'),
    avg_trade_size=(size_col,'mean'),
    avg_leverage=(leverage_col,'mean'),
    num_trades=('Closed PnL','count'),
    long_ratio=('Side', lambda x: (x=="Buy").mean())
).reset_index()

st.subheader("Sample Daily Metrics per Trader")
st.dataframe(daily_metrics.head())


st.subheader("Performance by Sentiment")
pnl_sentiment = df.groupby('classification')['Closed PnL'].mean()
st.write(pnl_sentiment)

fig1, ax1 = plt.subplots()
sns.barplot(x=pnl_sentiment.index, y=pnl_sentiment.values, ax=ax1, color="skyblue")
ax1.set_ylabel("Average PnL")
ax1.set_xlabel("Sentiment")
plt.xticks(rotation=45)
st.pyplot(fig1)

behavior_sentiment = df.groupby('classification').agg(
    avg_leverage=(leverage_col,'mean'),
    trade_count=('Closed PnL','count'),
    long_ratio=('Side', lambda x: (x=="Buy").mean()),
    avg_trade_size=(size_col,'mean')
)
st.subheader("Behavior by Sentiment")
st.dataframe(behavior_sentiment)


st.subheader("Trader Segmentation")

trade_counts = df.groupby(account_col)['Closed PnL'].count().reset_index()
trade_counts['frequency_segment'] = np.where(trade_counts['Closed PnL'] > trade_counts['Closed PnL'].median(),'Frequent','Infrequent')

avg_leverage = df.groupby(account_col)[leverage_col].mean().reset_index()
avg_leverage['leverage_segment'] = np.where(avg_leverage[leverage_col] > avg_leverage[leverage_col].median(),'High Leverage','Low Leverage')

segments = pd.merge(trade_counts[[account_col,'frequency_segment']], avg_leverage[[account_col,'leverage_segment']], on=account_col)
st.write("Sample Segmented Traders")
st.dataframe(segments.head())

segment_pnl = df.groupby(account_col)['Closed PnL'].sum().reset_index()
segment_pnl = pd.merge(segment_pnl, segments, on=account_col)
segment_summary = segment_pnl.groupby(['frequency_segment','leverage_segment'])['Closed PnL'].mean()
st.write("Average PnL by Trader Segment")
st.dataframe(segment_summary)


st.subheader("Actionable Insights / Strategy")
st.write("""
1️⃣ Reduce leverage during Extreme Fear days for all traders.  
2️⃣ Increase trading frequency during Extreme Greed days for frequent/high-leverage traders.  
3️⃣ Prioritize trades with higher win-rate accounts during Fear days.  
""")

st.subheader("Bonus: Trader Clustering (Behavioral Archetypes)")

clustering_data = df.groupby(account_col).agg(
    avg_PnL=('Closed PnL','mean'),
    avg_leverage=(leverage_col,'mean'),
    trade_count=('Closed PnL','count'),
    long_ratio=('Side', lambda x: (x=="Buy").mean())
).reset_index()

scaler = StandardScaler()
X = scaler.fit_transform(clustering_data[['avg_PnL','avg_leverage','trade_count','long_ratio']])

kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['cluster'] = kmeans.fit_predict(X)

st.write("Trader Clusters Sample")
st.dataframe(clustering_data.head())

fig2, ax2 = plt.subplots()
sns.countplot(x='cluster', data=clustering_data, ax=ax2, color="lightgreen")
ax2.set_title("Number of Traders per Cluster")
st.pyplot(fig2)

