import os
import subprocess

class Pipeline:
    def __init__(self, base_path, model_name):
        self.base_path = base_path
        self.model_name = model_name
        self.preprocessing_script = os.path.join(base_path, "a_data_preprocessing.py")
        self.feature_engineering_script = os.path.join(base_path, "b_feature_engineering.py")
        self.split_data_script = os.path.join(base_path, "c_split_data.py")
        self.train_model_script = os.path.join(base_path, "d_train_model.py")
        self.evaluate_model_script = os.path.join(base_path, "e_evaluate_model.py")
        self.load_and_predict_script = os.path.join(base_path, "f_load_and_predict.py")

    def run_script(self, script_name, args=[]):
        command = ["python", script_name] + args
        print(f"Running command: {command}")  # Added print to check the command being executed
        subprocess.run(command, check=True)

    def preprocess_data(self):
        print("Running data_preprocessing.py...")
        self.run_script(self.preprocessing_script)

    def engineer_features(self):
        print("Running feature_engineering.py...")
        self.run_script(self.feature_engineering_script)

    def split_data(self):
        print("Running split_data.py...")
        self.run_script(self.split_data_script)

    def train_model(self):
        print(f"Model name to train: {self.model_name}")  # Added print to check the model name
        print("Running train_model.py...")
        self.run_script(self.train_model_script, ["--model", self.model_name])

    def evaluate_model(self):
        print("Running evaluate_model.py...")
        self.run_script(self.evaluate_model_script, ["--model", self.model_name])

    def load_and_predict(self, forecast_period):
        print("Running load_and_predict.py...")
        self.run_script(self.load_and_predict_script, ["--model", self.model_name, "--forecast_period", str(forecast_period)])

    def run_pipeline(self, forecast_period): 
        self.preprocess_data()
        self.engineer_features()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.load_and_predict(forecast_period)
        print("Pipeline execution completed successfully.")


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# modelos testados e validados: "sarimax", "prophet", "naive_drift", "seasonal_naive", "holtwinters"

model_name = "prophet"  # prophet, naive_drift, seasonal_naive, sarimax, holtwinters *** ser o nome exato dos modelos ***
model_name = model_name.lower() # Lower para garantir que o nome do modelo permaneça em lowercase

forecast_period = 30  # 30 days to forecast - TESTE *** CHECAR SE O VALOR É UM INTEIRO POR EXEMPLO ***
pipeline = Pipeline(base_path, model_name)
pipeline.run_pipeline(forecast_period)