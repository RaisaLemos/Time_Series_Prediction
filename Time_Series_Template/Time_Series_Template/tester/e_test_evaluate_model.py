import unittest
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from darts.models import NaiveSeasonal, NaiveDrift
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import sys

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        try:
            # Definir o caminho do arquivo de dados
            data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'test_data_unit.csv'))
            
            # Carregar os dados a partir do arquivo CSV
            data = pd.read_csv(data_file_path)
            
            # Assegurar que as colunas necessárias estão presentes
            assert 'ds' in data.columns, "A coluna 'ds' está ausente no arquivo de dados."
            assert 'y' in data.columns, "A coluna 'y' está ausente no arquivo de dados."
            
            # Dividir os dados em treino e teste
            self.train_data = data.iloc[:-12]
            self.test_data = data.iloc[-12:]
        except Exception as e:
            self.fail(f"Erro ao configurar os dados de teste: {e}")

    def test_evaluate_sarimax(self):
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        model = SARIMAX(self.train_data['y'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)

        y_pred_train = fitted_model.fittedvalues
        y_pred_test = fitted_model.get_forecast(steps=len(self.test_data)).predicted_mean

        self._evaluate_predictions(y_pred_train, y_pred_test, 'SARIMAX')

    def test_evaluate_prophet(self):
        model = Prophet()
        model.fit(self.train_data)

        future = model.make_future_dataframe(periods=len(self.test_data))
        forecast = model.predict(future)

        y_pred_train = forecast['yhat'][:len(self.train_data)]
        y_pred_test = forecast['yhat'][-len(self.test_data):]

        self._evaluate_predictions(y_pred_train, y_pred_test, 'Prophet')

    def test_evaluate_holtwinters(self):
        model = ExponentialSmoothing(self.train_data['y'], trend='add', seasonal='add', seasonal_periods=12)
        fitted_model = model.fit()

        y_pred_train = fitted_model.fittedvalues[-len(self.train_data):]
        y_pred_test = fitted_model.forecast(len(self.test_data))

        self._evaluate_predictions(y_pred_train, y_pred_test, 'Holt-Winters')

    def test_evaluate_naive_seasonal(self):
        k = 12
        model = NaiveSeasonal(K=k)
        train_series = TimeSeries.from_series(self.train_data['y'])
        model.fit(train_series)

        y_pred_train = model.predict(len(self.train_data['y'])).values().flatten()
        y_pred_test = model.predict(len(self.test_data['y']), num_samples=1).values().flatten()

        self._evaluate_predictions(y_pred_train, y_pred_test, 'Naive Seasonal')

    def test_evaluate_naive_drift(self):
        model = NaiveDrift()
        train_series = TimeSeries.from_series(self.train_data['y'])
        model.fit(train_series)

        y_pred_train = model.predict(len(self.train_data['y'])).values().flatten()
        y_pred_test = model.predict(len(self.test_data['y'])).values().flatten()

        self._evaluate_predictions(y_pred_train, y_pred_test, 'Naive Drift')

    def _evaluate_predictions(self, y_pred_train, y_pred_test, model_name):
        # Calcular métricas de avaliação
        mae_train = mean_absolute_error(self.train_data['y'], y_pred_train)
        mae_test = mean_absolute_error(self.test_data['y'], y_pred_test)

        mape_train = mean_absolute_percentage_error(self.train_data['y'], y_pred_train) * 100
        mape_test = mean_absolute_percentage_error(self.test_data['y'], y_pred_test) * 100

        # Fazendo o RMSE na mão
        mse_train = mean_squared_error(self.train_data['y'], y_pred_train)
        mse_test = mean_squared_error(self.test_data['y'], y_pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)

        # Imprimir métricas
        print(f"Modelo {model_name} - Métricas avaliativas:")
        print("\n")
        print(f"Mean Absolute Percentage Error (MAPE) - Treino: {mape_train:.2f}, Teste: {mape_test:.2f}")
        print(f"Mean Absolute Error (MAE) - Treino: {mae_train:.2f}, Teste: {mae_test:.2f}")
        print(f"Root Mean Squared Error (RMSE) - Treino: {rmse_train:.2f}, Teste: {rmse_test:.2f}")
        print("\n")


unittest.main()