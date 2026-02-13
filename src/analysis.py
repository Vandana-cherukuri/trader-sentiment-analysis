import matplotlib.pyplot as plt
import seaborn as sns


def sentiment_performance_analysis(df, output_dir):

    result = df.groupby('classification')['Closed PnL'].mean()
    print("\nAverage PnL by Sentiment:\n", result)

    plt.figure()
    sns.barplot(x=result.index, y=result.values)
    plt.xticks(rotation=45)
    plt.title("Average PnL by Sentiment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_performance.png")
    plt.close()


def fear_vs_greed_analysis(df):

    summary = df.groupby('classification').agg({
        'Closed PnL': 'mean',
        'win': 'mean',
        'Size USD': 'mean',
        'leverage': 'mean',
        'Trade ID': 'count'
    })

    print("\n===== FEAR vs GREED PERFORMANCE =====")
    print(summary)


def behavior_by_sentiment(df):

    behavior = df.groupby('classification').agg({
        'Side': lambda x: (x == 'BUY').mean(),
        'leverage': 'mean',
        'Trade ID': 'count'
    })

    print("\n===== BEHAVIOR BY SENTIMENT =====")
    print(behavior)


def trader_segmentation(df, output_dir):

    trader_profile = df.groupby('Account').agg({
        'Closed PnL': 'mean',
        'Size USD': 'mean',
        'leverage': 'mean',
        'Trade ID': 'count',
        'win': 'mean'
    }).reset_index()

    trader_profile.rename(columns={
        'Closed PnL': 'avg_pnl',
        'Size USD': 'avg_size',
        'Trade ID': 'trade_count',
        'win': 'win_rate'
    }, inplace=True)

    trader_profile.to_csv(
        f"{output_dir}/trader_profiles.csv",
        index=False
    )

    return trader_profile

