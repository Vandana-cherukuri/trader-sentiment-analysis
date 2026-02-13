import pandas as pd


def load_data(sentiment_path, trades_path):
    sentiment = pd.read_csv(sentiment_path)
    trades = pd.read_csv(trades_path)
    return sentiment, trades


def data_quality_report(sentiment, trades):

    print("\n===== DATA QUALITY REPORT =====")

    print("\nSentiment Dataset")
    print("Rows:", sentiment.shape[0])
    print("Columns:", sentiment.shape[1])
    print("Missing:\n", sentiment.isnull().sum())
    print("Duplicates:", sentiment.duplicated().sum())

    print("\nTrades Dataset")
    print("Rows:", trades.shape[0])
    print("Columns:", trades.shape[1])
    print("Missing:\n", trades.isnull().sum())
    print("Duplicates:", trades.duplicated().sum())


def clean_data(sentiment, trades):
    sentiment = sentiment.dropna()
    trades = trades.dropna()
    return sentiment, trades


def merge_data(sentiment, trades):

    sentiment['date'] = pd.to_datetime(sentiment['date'])
    trades['Timestamp IST'] = pd.to_datetime(
        trades['Timestamp IST'],
        dayfirst=True
    )

    sentiment['date'] = sentiment['date'].dt.date
    trades['date'] = trades['Timestamp IST'].dt.date

    df = pd.merge(trades, sentiment, on='date', how='inner')

    return df


def engineer_features(df):

    df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce')
    df['Size USD'] = pd.to_numeric(df['Size USD'], errors='coerce')
    df['Execution Price'] = pd.to_numeric(df['Execution Price'], errors='coerce')

    if 'leverage' in df.columns:
        df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
    else:
        df['leverage'] = 1

    df['win'] = df['Closed PnL'] > 0

    daily_trader = df.groupby(['Account', 'date']).agg({
        'Closed PnL': 'sum',
        'Size USD': 'mean',
        'leverage': 'mean',
        'Trade ID': 'count',
        'win': 'mean'
    }).reset_index()

    daily_trader.rename(columns={
        'Closed PnL': 'daily_pnl',
        'Size USD': 'avg_trade_size',
        'Trade ID': 'trades_per_day',
        'win': 'win_rate'
    }, inplace=True)

    return df, daily_trader
