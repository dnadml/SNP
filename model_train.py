from math import sqrt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_train_get_data import prep_data

def train_and_evaluate(X, y):
    # Training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=150)
    # model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.05, max_depth=7, alpha=10, n_estimators=200)

    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_mean_rmse = np.mean(cv_rmse)

    print(f"Cross-validation RMSE: {cv_mean_rmse}")

    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, scaler, {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "CV_RMSE": cv_mean_rmse
    }

def plot_feature_importance(model, features, interval):
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Create a DataFrame for easier handling
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance for {interval} Interval')
    plt.gca().invert_yaxis()  # Invert axis to have the most important at the top
    plt.tight_layout()

    # plt.show()


def create_models():
    data_dfs = prep_data()
    metrics = {}

    for interval, data_df in data_dfs.items():
        print(f"Training model for {interval} data...")
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df.dropna(inplace=True)
        data_df = data_df.select_dtypes(include=[np.number])

        feature_columns = [col for col in data_df.columns if 'NextClose' not in col]
        X = data_df[feature_columns]
        y = data_df['NextClose']

        # Train and evaluate
        model, scaler, result = train_and_evaluate(X, y)
        metrics[interval] = result
        print(f"Results for {interval} interval: {result}")

         # Plot feature importance
        plot_feature_importance(model, feature_columns, interval)

        # Save model and scaler
        joblib.dump(model, f'/snpOracle/SNP/model_gb_{interval}.pkl')
        joblib.dump(scaler, f'/snpOracle/SNP/scaler_gb_{interval}.pkl')

    return metrics

def print_metrics(metrics):
    print("\nSummary of Model Performance:")
    print(f"{'Interval':<10}{'MSE':>10}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for interval, metric in metrics.items():
        mse = f"{metric['MSE']:.2f}"
        rmse = f"{metric['RMSE']:.2f}"
        mae = f"{metric['MAE']:.2f}"
        r2 = f"{metric['R2']:.4f}"
        print(f"{interval:<10}{mse:>10}{rmse:>10}{mae:>10}{r2:>10}")

if __name__ == '__main__':
    model_metrics = create_models()
    print_metrics(model_metrics)
