from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def clustering(trader_profile, n_clusters, output_dir):

    features = trader_profile[['avg_pnl', 'avg_size', 'trade_count', 'win_rate']]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    trader_profile['cluster'] = kmeans.fit_predict(features)

    print("\nCluster Summary:")
    print(
        trader_profile
        .groupby('cluster')[['avg_pnl', 'avg_size', 'trade_count', 'win_rate']]
        .mean()
    )

    trader_profile.to_csv(
        f"{output_dir}/trader_clusters.csv",
        index=False
    )

    return trader_profile


def predictive_model(df, test_size, random_state):

    print("\n===== NEXT-DAY PROFITABILITY PREDICTION =====")

    # Sort properly
    df = df.sort_values(['Account', 'date'])

    # Next-day PnL
    df['next_day_pnl'] = df.groupby('Account')['Closed PnL'].shift(-1)

    # Create profitability bucket
    df['profit_bucket'] = (df['next_day_pnl'] > 0).astype(int)

    df = df.dropna()

    X = df[['value', 'Size USD', 'leverage', 'win']]
    y = df['profit_bucket']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model

