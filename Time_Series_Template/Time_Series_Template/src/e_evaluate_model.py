import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from prophet import Prophet
from darts import TimeSeries
from darts.models import NaiveSeasonal, NaiveDrift
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import argparse
import os

def evaluate_model(model, train_data, test_data):
    # Fazer previsões
    if isinstance(model, SARIMAXResultsWrapper):
        y_pred_train = model.fittedvalues
        y_pred_test = model.get_forecast(steps=len(test_data)).predicted_mean
    elif isinstance(model, ExponentialSmoothing):
        model_fit = model.fit()
        y_pred_train = model_fit.fittedvalues[-len(train_data):]
        y_pred_test = model_fit.forecast(len(test_data))
    elif isinstance(model, Prophet):
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        y_pred_train = forecast['yhat'][:len(train_data)]
        y_pred_test = forecast['yhat'][-len(test_data):]
    elif isinstance(model, NaiveSeasonal):
        train_series = TimeSeries.from_series(train_data['y'])
        model.fit(train_series)
        y_pred_test = model.predict(len(test_data['y']), num_samples=1).values().flatten()
        y_pred_train = model.predict(len(train_data['y'])).values().flatten()
    elif isinstance(model, NaiveDrift):
        y_pred_train = model.predict(len(train_data)).values()[:len(train_data)]
        y_pred_test = model.predict(len(test_data)).values()
    else:
        # Para Holt-Winters, usar ExponentialSmoothing diretamente
        try:
            model = ExponentialSmoothing(train_data['y'], trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            y_pred_train = fitted_model.fittedvalues[-len(train_data):]
            y_pred_test = fitted_model.forecast(len(test_data))
        except Exception as e:
            raise ValueError(f'Erro ao avaliar modelo Holt-Winters: {str(e)}')

    # Verificar comprimento das previsões
    print(f"Comprimento dos dados de treino: {len(train_data['y'])}, Comprimento das previsões de treino: {len(y_pred_train)}")
    print(f"Comprimento dos dados de teste: {len(test_data['y'])}, Comprimento das previsões de teste: {len(y_pred_test)}")

    # Calcular métricas de avaliação
    mae_train = mean_absolute_error(train_data['y'], y_pred_train)
    mae_test = mean_absolute_error(test_data['y'], y_pred_test)

    mape_train = mean_absolute_percentage_error(train_data['y'], y_pred_train) * 100
    mape_test = mean_absolute_percentage_error(test_data['y'], y_pred_test) * 100

    mse_train = mean_squared_error(train_data['y'], y_pred_train)
    mse_test = mean_squared_error(test_data['y'], y_pred_test)
    
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    # Imprimir métricas
    print(f"Modelo {args.model} - Métricas avaliativas:")
    print("\n")
    print(f"Mean Absolute Percentage Error (MAPE) - Treino: {mape_train:.2f}, Teste: {mape_test:.2f}")
    print(f"Mean Absolute Error (MAE) - Treino: {mae_train:.2f}, Teste: {mae_test:.2f}")
    print(f"Root Mean Squared Error (RMSE) - Treino: {rmse_train:.2f}, Teste: {rmse_test:.2f}")
    print("\n")

    # UTILIZAR A PARTE DO GRÁFICO CASO QUEIRA
    # Plotar previsões vs dados reais para o conjunto de teste
    plt.figure(figsize=(14, 8))
    plt.plot(train_data['ds'], train_data['y'], label='Treino')
    plt.plot(test_data['ds'], test_data['y'], label='Teste')
    plt.plot(test_data['ds'], y_pred_test, label='Previsão')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.title(f'Previsão vs Dados Reais - Modelo {args.model}')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_model(model_path, model_name):
    model_path = rf"{model_path}/{model_name}.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="O modelo para avaliar")
args = parser.parse_args()

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model = load_model(model_path, args.model)

# Load pre-divided training and test data
train_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'train_data.csv')))
test_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model_data_train_test', 'test_data.csv')))

evaluate_model(model, train_data, test_data)

print("Avaliação do modelo concluída com sucesso.")
print("\n")