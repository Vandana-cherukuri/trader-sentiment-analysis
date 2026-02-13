# Trader Sentiment Analysis Dashboard

This project analyzes the relationship between Bitcoin market sentiment (Fear/Greed) and trader performance on Hyperliquid. The dashboard allows you to explore trader behavior, PnL, and actionable insights for smarter trading strategies.

#  Features
- Daily PnL per trader
- Win rate, trade frequency, trade size, long/short ratio
- Sentiment-based performance analysis
- Trader segmentation (frequency & leverage)
- Behavioral clustering of traders
- Actionable strategy recommendations



# Setup

1. **Clone the repo**:

git clone https://github.com/Vandana-cherukuri/trader-sentiment-analysis
cd trader-sentiment-analysis
Install dependencies:

pip install -r requirements.txt
Make sure the CSV files are in the data/ folder:

fear_greed.csv

trade.csv

Run the Streamlit dashboard:
streamlit run app.py

