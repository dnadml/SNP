from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SNP.get_data_prod import prep_data

def create_model():
    data_df = prep_data(drop_na=True) 
    X = data_df.select_dtypes(include=[np.number]).drop(columns=['NextClose'])
    y = data_df['NextClose']

    # training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale data on trained and test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # initialize and train the gradient boosting regressor
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42) # Original
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, min_samples_split=4, min_samples_leaf=1, max_depth=6, max_features=None) # Hypertuned
    model.fit(X_train_scaled, y_train)

    # cv training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=6, scoring='neg_mean_squared_error')

    # get CV metrics
    cv_rmse_scores = np.sqrt(-cv_scores)
    print(f"CV RMSE Scores: {cv_rmse_scores}")
    print(f"CV RMSE Mean: {np.mean(cv_rmse_scores)}")
    print(f"CV RMSE Standard Deviation: {np.std(cv_rmse_scores)}")

    #### PLOT ####
    # get feature importance
    feature_importance = model.feature_importances_
    # dataframe for plot
    features = pd.Series(feature_importance, index=X_train.columns).sort_values(ascending=False)
    # plot
    plt.figure(figsize=(20,12))
    features.plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.show()
    
    # predict and evaluate
    predictions = model.predict(X_test_scaled)

    # metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # print metrics
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # save model and scale for predict
    joblib.dump(model, './snpOracle/SNP/model_gb.pkl')
    joblib.dump(scaler, './snpOracle/SNP/scaler_gb.pkl')

if __name__ == '__main__':
    create_model()

############################# Testing Hyper Parameters Random ##########################################

# def create_model():
#     data_df = prep_data() 
#     X = data_df.select_dtypes(include=[np.number]).drop(columns=['NextClose'])
#     y = data_df['NextClose']


#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # parameters for RandomizedSearchCV
#     param_distributions = {
#         'n_estimators': [100, 200, 300, 400, 500],
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],
#         'max_depth': [3, 4, 5, 6, 7],
#         'min_samples_split': [2, 4, 6],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': [None, 'sqrt', 'log2'],
#     }

#     # initialize the base model for tuning
#     base_model = GradientBoostingRegressor(random_state=42)

#     # initialize RandomizedSearchCV 
#     rnd_search = RandomizedSearchCV(base_model, param_distributions=param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

#     # fit RandomizedSearchCV
#     rnd_search.fit(X_train_scaled, y_train) 

#     # printing best parameters
#     print("Best parameters found: ", rnd_search.best_params_)  

#     # use best model
#     best_model = rnd_search.best_estimator_  # Get the best model
#     predictions = best_model.predict(X_test_scaled)  # Predictions with the best model

#     # metric calculations
#     mse = mean_squared_error(y_test, predictions)
#     rmse = sqrt(mse)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)

#     # print metrics
#     print(f"Mean Squared Error (MSE): {mse:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"R-squared (R2): {r2:.2f}")

#     # save jobs
#     # joblib.dump(best_model, './mining_models/best_model_gb.pkl')
#     # joblib.dump(scaler, './mining_models/scaler_gb.pkl')

# if __name__ == '__main__':
#     create_model()



# ############################### Testing Hyper Parameters Grid Search ##########################################
# def create_model():
#     data_df = prep_data(drop_na=True) 
    
#     X = data_df.select_dtypes(include=[np.number]).drop(columns=['NextClose'])
#     y = data_df['NextClose']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

    # # define parameter RandomizedSearchCV
    # param_grid = {
        # 'n_estimators': [150, 200, 250],
        # 'learning_rate': [0.01, 0.05, 0.1],
        # 'max_depth': [2, 3, 4],
        # 'min_samples_split': [2, 4, 6],
        # 'min_samples_leaf': [1, 2, 3],
        # 'max_features': [None, 'sqrt', 'log2'],    
    # }

#     # initialize base model for GridSearchCV
#     base_model = GradientBoostingRegressor(random_state=42)
    
#     # initialize GridSearchCV parameter grid
#     grid_search = GridSearchCV(base_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

#     # fit GridSearchCV
#     grid_search.fit(X_train_scaled, y_train)

#     # print the best parameters
#     print("Best parameters found: ", grid_search.best_params_)

#     # use best model for predictions
#     best_model = grid_search.best_estimator_
#     predictions = best_model.predict(X_test_scaled)
    
#     # metrics
#     mse = mean_squared_error(y_test, predictions)
#     rmse = sqrt(mse)
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)

#     # print metrics
#     print(f"Mean Squared Error (MSE): {mse:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"R-squared (R2): {r2:.2f}")

#     # save jobs
#     # joblib.dump(best_model, './mining_models/best_model_gb.pkl')
#     # joblib.dump(scaler, './mining_models/scaler_gb.pkl')

# if __name__ == '__main__':
#     create_model()
