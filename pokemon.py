# pokemon.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    pokemon_stats = pd.read_csv("pokemon_summary.csv")
    trainers_df   = pd.read_csv("trainers.csv")

    tp = trainers_df.copy()
    stats_cols = ['attack','defense','hp','sp_attack','sp_defense','speed']
    for i in range(1,7):
        slot = f'pokemon{i}'
        block = (
            pokemon_stats[['name'] + stats_cols]
            .rename(columns={c: f"{c}{i}" for c in stats_cols})
            .rename(columns={'name': slot})
        )
        tp = tp.merge(block, on=slot, how='left')
    
    feature_cols = [f"{stat}{i}" for i in range(1,7) for stat in stats_cols] + \
                   [f"level_pokemon{i}" for i in range(1,7)]
    X = tp[feature_cols].fillna(0).values
    y = tp['total_EXP'].fillna(0).values


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1
    )

    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr',     LinearRegression())
        ]),
        'Polynomial Regression (deg=2)': Pipeline([
            ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lr',     LinearRegression())
        ]),
        'Ridge Regression (deg=2)': Pipeline([
            ('poly',    PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler',  StandardScaler()),
            ('ridge',   Ridge(alpha=1.0, random_state=42))
        ]),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)
        results.append({
            'Model':     name,
            'Train MSE': mean_squared_error(y_train, y_tr),
            'Train R²':  r2_score(y_train, y_tr),
            'Test MSE':  mean_squared_error(y_test,  y_te),
            'Test R²':   r2_score(y_test,  y_te),
        })
    metrics_df = pd.DataFrame(results)

    print("\n=== Regression Model Performance ===")
    print(metrics_df.to_string(index=False, float_format='%.4f'))

    plt.figure(figsize=(6,4))
    plt.bar(metrics_df['Model'], metrics_df['Test R²'])
    plt.ylabel('Test R²')
    plt.title('Test R² Comparison')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.bar(metrics_df['Model'], metrics_df['Test MSE'])
    plt.ylabel('Test MSE')
    plt.title('Test MSE Comparison')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,6))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.5, label=name)
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], '--', color='gray')
    plt.xlabel('Actual EXP')
    plt.ylabel('Predicted EXP')
    plt.title('Predicted vs Actual EXP (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
