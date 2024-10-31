import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper, SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from darts.models import NaiveSeasonal, NaiveDrift
import argparse
import os

def load_and_predict(model_info, new_data, forecast_period):
    # Make predictions
    if isinstance(model_info, Prophet):
        # Prophet model
        future = model_info.make_future_dataframe(periods=forecast_period, include_history=False)
        forecast = model_info.predict(future)
        y_pred = forecast['yhat'].values[-forecast_period:]
    
    elif hasattr(model_info, 'model') and isinstance(model_info.model, ExponentialSmoothing):
        # Para Holt-Winters, usamos o modelo diretamente
        y_pred = model_info.forecast(forecast_period)
    
    elif isinstance(model_info, (NaiveSeasonal, NaiveDrift)):
        # Darts models
        y_pred = list(model_info.predict(forecast_period).values())
    
    elif 'order' in model_info and 'seasonal_order' in model_info:
        # SARIMAX model
        order = model_info['order']
        seasonal_order = model_info['seasonal_order']
        sarimax_model = SARIMAX(new_data['y'], order=order, seasonal_order=seasonal_order)
        fitted_model = model_info['model']  # SARIMAXResultsWrapper object

        # Forecast using the fitted model
        y_pred = fitted_model.forecast(steps=forecast_period)
    
    else:
        raise ValueError(f'Model {type(model_info)} not supported for prediction.')

    return y_pred

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="The model to use for prediction")
parser.add_argument('--forecast_period', type=int, required=True, help="Number of days to forecast")
args, unknown = parser.parse_known_args()

def load_model(model_path, model_name):
    model_path = rf"{model_path}/{model_name}.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model = load_model(model_path, args.model)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Processed_data' , 'processed_data.csv'))

print(f"Trying to load data file from: {data_path}")
new_data = pd.read_csv(data_path)

forecast_period = int(args.forecast_period)

try:
    predictions = load_and_predict(model, new_data, forecast_period)
except ValueError as ve:
    print(f'Error during prediction: {str(ve)}')
    raise ve

predictions_df = pd.DataFrame({'ds': new_data['ds'].values[-forecast_period:], 'y_pred': predictions})

print("Model predictions:")
print(predictions_df)

output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_output_prediction', f'predictions_{args.model}_{args.forecast_period}_days.csv'))
predictions_df.to_csv(output_file, index=False)

print(f"Predictions generated and saved successfully at {output_file}")