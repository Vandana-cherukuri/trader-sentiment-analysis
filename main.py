from src.utils import load_config, ensure_directory
from src.data_processing import (
    load_data,
    data_quality_report,
    clean_data,
    merge_data,
    engineer_features
)
from src.analysis import (
    sentiment_performance_analysis,
    fear_vs_greed_analysis,
    behavior_by_sentiment,
    trader_segmentation
)
from src.modeling import clustering, predictive_model


def main():

    config = load_config("config.yaml")
    ensure_directory(config['output']['output_dir'])

    # Load data
    sentiment, trades = load_data(
        config['data']['sentiment_path'],
        config['data']['trades_path']
    )

    # Data quality report
    data_quality_report(sentiment, trades)

    # Clean
    sentiment, trades = clean_data(sentiment, trades)

    # Merge
    df = merge_data(sentiment, trades)

    # Feature engineering
    df, daily_trader = engineer_features(df)

    # Analysis
    sentiment_performance_analysis(df, config['output']['output_dir'])
    fear_vs_greed_analysis(df)
    behavior_by_sentiment(df)

    # Trader segmentation
    trader_profile = trader_segmentation(df, config['output']['output_dir'])

    # Clustering
    trader_profile = clustering(
        trader_profile,
        config['model']['n_clusters'],
        config['output']['output_dir']
    )

    # Predictive model
    predictive_model(
        df,
        config['model']['test_size'],
        config['model']['random_state']
    )

    print("\n===== STRATEGY RECOMMENDATIONS =====")
    print("""
    1. Reduce leverage during Extreme Fear regimes.
    2. Allow controlled frequency increase during Greed for high win-rate traders.
    """)


if __name__ == "__main__":
    main()
