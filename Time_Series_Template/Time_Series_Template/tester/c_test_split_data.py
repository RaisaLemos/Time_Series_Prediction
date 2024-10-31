import unittest
import pandas as pd
import sys
import os

# Adicione o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.c_split_data as c_split_data   # Importar a função de preprocessamento

class TestSplitData(unittest.TestCase):

    def setUp(self):
        try:
            # Definir o caminho do arquivo de dados
            data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tester_data', 'test_data_unit.csv'))
            
            # Carregar os dados a partir do arquivo CSV
            self.data_sample = pd.read_csv(data_file_path)
            
            # Assegurar que as colunas necessárias estão presentes
            assert 'ds' in self.data_sample.columns, "A coluna 'ds' está ausente no arquivo de dados."
            assert 'y' in self.data_sample.columns, "A coluna 'y' está ausente no arquivo de dados."
        
        except Exception as e:
            print(f"Erro ao configurar o teste: {e}")
            self.fail(f"Falha na configuração do teste devido ao erro: {e}")
    
    def test_split(self):
        # Testar função de divisão de dados
        train_data, test_data = c_split_data.split_data_for_time_series(self.data_sample, test_size=0.2)
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        self.assertEqual(len(train_data) + len(test_data), len(self.data_sample))  # Verificar se o tamanho é mantido


unittest.main()