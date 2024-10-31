import unittest
import pandas as pd
import os
import pickle

class TestLoadAndPredict(unittest.TestCase):

    def setUp(self):
        try:
            # Define the path to the data file
            data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'test_data_unit.csv'))
            
            # Load the data from the CSV file
            self.data = pd.read_csv(data_file_path)
            
            # Ensure necessary columns are present
            assert 'ds' in self.data.columns, "Column 'ds' is missing from the data file."
            assert 'y' in self.data.columns, "Column 'y' is missing from the data file."
            
            # Define the forecast period
            self.forecast_period = 12
            
            # Path to the directory where models are saved
            self.models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tester_models'))
        except Exception as e:
            self.fail(f"Error setting up test data: {e}")

    def load_model(self, model_name):
        model_path = os.path.join(self.models_path, f"{model_name}.pkl")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def predict_sarimax(self, model_info):
        order = model_info['order']
        seasonal_order = model_info['seasonal_order']
        fitted_model = model_info['model']
        y_pred = fitted_model.forecast(steps=self.forecast_period)
        return y_pred

    def predict_prophet(self, model_info):
        future = model_info.make_future_dataframe(periods=self.forecast_period, include_history=False)
        forecast = model_info.predict(future)
        y_pred = forecast['yhat'].values[-self.forecast_period:]
        return y_pred

    def predict_holtwinters(self, model_info):
        y_pred = model_info.forecast(self.forecast_period)
        return y_pred

    def predict_naive_seasonal(self, model_info):
        y_pred = list(model_info.predict(self.forecast_period).values())
        return y_pred

    def predict_naive_drift(self, model_info):
        y_pred = list(model_info.predict(self.forecast_period).values())
        return y_pred

    def test_load_and_predict_sarimax(self):
        model_info = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12),
            'model': self.load_model('sarimax_model')
        }
        y_pred = self.predict_sarimax(model_info)
        self.assertEqual(len(y_pred), self.forecast_period)

    def test_load_and_predict_prophet(self):
        model_info = self.load_model('prophet_model')
        y_pred = self.predict_prophet(model_info)
        self.assertEqual(len(y_pred), self.forecast_period)

    def test_load_and_predict_holtwinters(self):
        model_info = self.load_model('holtwinters_model')
        y_pred = self.predict_holtwinters(model_info)
        self.assertEqual(len(y_pred), self.forecast_period)

    def test_load_and_predict_naive_seasonal(self):
        model_info = self.load_model('naive_seasonal_model')
        y_pred = self.predict_naive_seasonal(model_info)
        self.assertEqual(len(y_pred), self.forecast_period)

    def test_load_and_predict_naive_drift(self):
        model_info = self.load_model('naive_drift_model')
        y_pred = self.predict_naive_drift(model_info)
        self.assertEqual(len(y_pred), self.forecast_period)


unittest.main()