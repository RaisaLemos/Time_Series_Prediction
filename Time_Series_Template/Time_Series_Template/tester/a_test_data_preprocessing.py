import unittest
import pandas as pd
import sys
import os

# Adicione o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.a_data_preprocessing as a_data_preprocessing   # Importar a função de preprocessamento

class TestDataPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Definir caminhos para os dados de teste
        test_raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'test_data_unit.csv'))
        test_processed_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'processed_test_data_unit.csv'))

        # Chamar função de preprocessamento nos dados de teste
        a_data_preprocessing.preprocess_data(test_raw_data_path, test_processed_data_path)

        # Carregar dados processados
        processed_data = pd.read_csv(test_processed_data_path)

        # Verificar se as colunas 'ds' e 'y' existem
        self.assertIn('ds', processed_data.columns)
        self.assertIn('y', processed_data.columns)

        # Verificar se 'ds' pode ser convertido para datetime
        try:
            pd.to_datetime(processed_data['ds'], errors='raise')
        except ValueError:
            self.fail("Column 'ds' could not be converted to datetime")

        # Verificar se 'y' é numérico
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_data['y']))

        # Remover arquivos de teste criados
        # os.remove(test_raw_data_path)
        os.remove(test_processed_data_path)

unittest.main()