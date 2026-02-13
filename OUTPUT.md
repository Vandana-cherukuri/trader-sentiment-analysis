# Trader Performance vs Market Sentiment

This document summarizes the methodology, key insights, and actionable strategy 



## 1 Methodology

1. **Data Preparation**
   - Loaded `fear_greed.csv` and `trades.csv`.
   - Checked for missing values and duplicates.
   - Converted timestamps to datetime and merged trades with sentiment data by date.

2. **Feature Engineering**
   - Calculated daily metrics per trader:
     - Daily PnL
     - Win rate
     - Average trade size
     - Leverage (default 1 if missing)
     - Number of trades
     - Long/Short ratio
   - Segmented traders by:
     - Frequency: Frequent vs Infrequent
     - Leverage: High vs Low

3. **Analysis**
   - Compared performance and behavior across sentiment categories (Fear, Greed, Neutral).
   - Identified behavioral archetypes using clustering.
   - Visualized insights through charts and tables.

---

## 2️ Key Insights

1. **Performance Differences**
   - Extreme Greed days show the highest average PnL.
   - Extreme Fear and Neutral days have lower profitability.

2. **Behavior Changes by Sentiment**
   - Traders slightly increase Buy trades during Fear days.
   - Frequent/high-leverage traders trade more during Greed days.
   - Leverage usage is mostly stable except during extreme events.

3. **Trader Segmentation**
   - Frequent, high-leverage traders are more profitable.
   - Infrequent, low-leverage traders have lower PnL.
   - Clustering identifies three behavioral archetypes:
     1. Consistent high-PnL traders
     2. Frequent moderate-PnL traders
     3. Infrequent, low-PnL traders

---

## 3️ Strategy Recommendations

1. **During Extreme Fear**
   - Reduce leverage for all traders to limit risk.
   - Focus on trades from consistent winners to maintain profitability.

2. **During Extreme Greed**
   - Increase trade frequency for frequent/high-leverage traders.

3. **General**
   - Use behavioral archetypes to tailor strategies.
   - Track sentiment daily to adjust trade size, leverage, and frequency.

---

## 4️ Output Charts / Tables

Available in the dashboard:

- **Average PnL by Sentiment** (Bar chart)
- **Behavior by Sentiment** (Table: leverage, trade count, long ratio, avg trade size)
- **Trader Segmentation Table
- **Trader Cluster Visualization (Count plot per cluster)

> Charts can be saved as PNG for submission.
