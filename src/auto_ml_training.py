import pandas as pd
import numpy as np
from pycaret.regression import *
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    real_data = pd.read_csv('data/simulation_data.csv')
    preprocessed_data = pd.read_csv('data/preprocessed_data.csv')

    train_data = preprocessed_data.iloc[:15]
    test_data = real_data.iloc[15:20]

    reg = setup(data=train_data, target='x', session_id=123)

    best_model = compare_models()

    final_model = finalize_model(best_model)

    future_predictions = []
    actual_positions = test_data['x'].values

    for i in range(len(test_data)):
        current_test_data = test_data.iloc[:i+1].copy()
        
        if len(future_predictions) > 0:
            current_test_data.iloc[i, current_test_data.columns.get_loc('x')] = future_predictions[-1]

        future_prediction = predict_model(final_model, data=current_test_data)
        
        if 'Label' in future_prediction.columns:
            future_predictions.append(future_prediction['Label'].values[-1])
        else:
            future_predictions.append(future_prediction.iloc[:, -1].values[-1])

    future_predictions = np.array(future_predictions)
    mse = mean_squared_error(actual_positions, future_predictions)
    r2 = r2_score(actual_positions, future_predictions)

    print("Tahmin Edilen Gelecek Konumlar (X koordinatı):", future_predictions)
    print("Gerçek Konumlar (X koordinatı):", actual_positions)
    print(f"Mean Squared Error for future predictions: {mse}")
    print(f"R^2 Score for future predictions: {r2}")
