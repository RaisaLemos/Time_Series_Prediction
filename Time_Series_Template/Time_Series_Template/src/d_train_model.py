import pandas as pd
import pickle
import itertools
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from darts.models import NaiveSeasonal, NaiveDrift
from darts import TimeSeries
import argparse
import os

def train_model(train_data, model_name):
    print(f"Starting training for model: {model_name}")

    if model_name == 'sarimax':
        # SARIMAX tuning
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
        
        best_score = float('inf')
        best_params = None
        
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                try:
                    model = SARIMAX(train_data['y'], order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
                    fitted_model = model.fit(disp=False)
                    mae = mean_absolute_error(train_data['y'], fitted_model.fittedvalues)
                    if mae < best_score:
                        best_score = mae
                        best_params = (param, seasonal_param)
                except:
                    continue
        
        model = SARIMAX(train_data['y'], order=best_params[0], seasonal_order=best_params[1], enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit()

        # Save SARIMAX model information including parameters
        model_info = {
            'model': fitted_model,
            'order': best_params[0],
            'seasonal_order': best_params[1]
        }

    elif model_name == 'prophet':
        # Prophet tuning
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0]
        }
        
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        best_score = float('inf')
        best_params = None
        
        for params in all_params:
            model = Prophet(**params)
            model.fit(train_data)
            forecast = model.predict(train_data[['ds']])
            mae = mean_absolute_error(train_data['y'], forecast['yhat'])
            if mae < best_score:
                best_score = mae
                best_params = params
        
        model = Prophet(**best_params)
        model.fit(train_data)
        fitted_model = model

    elif model_name == 'holtwinters':
        # Holt-Winters tuning
        trend_options = ['add', 'mul', None]
        seasonal_options = ['add', 'mul', None]
        seasonal_periods = [12]

        best_score = float('inf')
        best_params = None
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                for seasonal_period in seasonal_periods:
                    try:
                        model = ExponentialSmoothing(train_data['y'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period)
                        fitted_model = model.fit()
                        mae = mean_absolute_error(train_data['y'], fitted_model.fittedvalues)
                        if mae < best_score:
                            best_score = mae
                            best_params = (trend, seasonal, seasonal_period)
                    except:
                        continue
        
        model = ExponentialSmoothing(train_data['y'], trend=best_params[0], seasonal=best_params[1], seasonal_periods=best_params[2])
        fitted_model = model.fit()

    elif model_name == 'seasonal_naive':
        # Seasonal Naive tuning
        best_score = float('inf')
        best_k = None
        
        for k in range(1, 25):
            model = NaiveSeasonal(K=k)
            fitted_model = model.fit(TimeSeries.from_series(train_data['y']))
            y_pred = fitted_model.historical_forecasts(TimeSeries.from_series(train_data['y']), start=len(train_data['y']) - k).values()
            mae = mean_absolute_error(train_data['y'][-k:], y_pred)
            if mae < best_score:
                best_score = mae
                best_k = k
        
        model = NaiveSeasonal(K=best_k)
        fitted_model = model.fit(TimeSeries.from_series(train_data['y']))

    elif model_name == 'naive_drift':
        # Naive Drift model
        model = NaiveDrift()
        fitted_model = model.fit(TimeSeries.from_series(train_data['y']))

    else:
        raise ValueError(f'Model {model_name} not recognized.')

    print(f"Finished training for model: {model_name}")
    return model_info if model_name == 'sarimax' else fitted_model

def save_model(model, model_name, path_to_save):
    path_to_save = rf"{path_to_save}/{model_name}.pkl"
    with open(path_to_save, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model {model_name} saved successfully at {path_to_save}")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="The model to train")
args = parser.parse_args()

# Load pre-divided training and test data
train_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'train_data.csv')))
test_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'test_data.csv')))

print(f"Training with model: {args.model}")

# Train the model
trained_model = train_model(train_data, args.model)

# Path to save the model in pickle
path_to_save = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Save the trained model
save_model(trained_model, args.model, path_to_save)

print("Training and saving the model completed successfully.")