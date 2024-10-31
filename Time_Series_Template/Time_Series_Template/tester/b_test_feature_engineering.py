# tests/test_feature_engineering.py
import unittest
import sys
import os

# Adicione o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.b_feature_engineering as b_feature_engineering   # Importar a função de feature_engineering

# Adicionar testes de feature engineering conforme necessidade
class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Setup inicial para dados de teste, se necessário
        pass

    def test_feature_engineering_function(self):
        # Testar as funções de engenharia de características
        pass


unittest.main()