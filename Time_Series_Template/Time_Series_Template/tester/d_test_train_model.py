import unittest
import pandas as pd
import sys
import os
import pickle

class TestTrainModel(unittest.TestCase):
    
    def setUp(self):
        try:
            # Definir o caminho do arquivo de dados
            data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'test_data_unit.csv'))
            
            # Carregar os dados a partir do arquivo CSV
            self.train_data = pd.read_csv(data_file_path)
            
            # Assegurar que as colunas necessárias estão presentes
            assert 'ds' in self.train_data.columns, "A coluna 'ds' está ausente no arquivo de dados."
            assert 'y' in self.train_data.columns, "A coluna 'y' está ausente no arquivo de dados."
            
            # Definir o caminho do diretório para salvar os modelos
            self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tester_models'))
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            self.fail(f"Erro ao configurar os dados de teste: {e}")
    
    def save_model(self, model, model_name):
        model_path = os.path.join(self.save_path, f"{model_name}.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    
    def test_train_sarimax(self):
        # Bloco de código para SARIMAX
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        
        model = SARIMAX(self.train_data['y'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        
        # Asserção para verificar se o modelo foi treinado corretamente
        self.assertIsNotNone(fitted_model)
        self.assertEqual(fitted_model.model.order, order)
        self.assertEqual(fitted_model.model.seasonal_order, seasonal_order)
        
        # Salvar o modelo treinado
        self.save_model(fitted_model, 'sarimax_model')
    
    def test_train_prophet(self):
        # Bloco de código para Prophet
        from prophet import Prophet
        
        model = Prophet()
        fitted_model = model.fit(self.train_data)
        
        # Asserção para verificar se o modelo foi treinado corretamente
        self.assertIsNotNone(fitted_model)
        
        # Salvar o modelo treinado
        self.save_model(fitted_model, 'prophet_model')
    
    def test_train_holtwinters(self):
        # Bloco de código para Holt-Winters
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        trend = 'add'
        seasonal = 'add'
        seasonal_periods = 12
        
        model = ExponentialSmoothing(self.train_data['y'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        fitted_model = model.fit()
        
        # Asserção para verificar se o modelo foi treinado corretamente
        self.assertIsNotNone(fitted_model)
        
        # Salvar o modelo treinado
        self.save_model(fitted_model, 'holtwinters_model')
    
    def test_train_seasonal_naive(self):
        # Bloco de código para Seasonal Naive
        from darts.models import NaiveSeasonal
        from darts import TimeSeries
        
        k = 12
        model = NaiveSeasonal(K=k)
        fitted_model = model.fit(TimeSeries.from_series(self.train_data['y']))
        
        # Asserção para verificar se o modelo foi treinado corretamente
        self.assertIsNotNone(fitted_model)
        
        # Salvar o modelo treinado
        self.save_model(fitted_model, 'naive_seasonal_model')
    
    def test_train_naive_drift(self):
        # Bloco de código para Naive Drift
        from darts.models import NaiveDrift
        from darts import TimeSeries
        
        model = NaiveDrift()
        fitted_model = model.fit(TimeSeries.from_series(self.train_data['y']))
        
        # Asserção para verificar se o modelo foi treinado corretamente
        self.assertIsNotNone(fitted_model)
        
        # Salvar o modelo treinado
        self.save_model(fitted_model, 'naive_drift_model')


unittest.main()