import pandas as pd
import os

def preprocess_data(raw_data_path, processed_data_path):
    # Carregar dados brutos
    data = pd.read_csv(raw_data_path, sep=",")

    # Realizar preprocessamento necessário
    # Exemplo: preenchimento de valores ausentes com forward fill
    data.ffill(inplace=True)

    # Alterar conforme necessidade, porém tentar manter o padrão ds e y
    data.rename(columns={"Date":"ds", "Daily minimum temperatures in Melbourne":"y"}, inplace=True)

    # Confirmando que a coluna de data está no padrão de data
    data["ds"] = pd.to_datetime(data["ds"], errors='coerce')
    data = data.sort_values(by=["ds"], ascending=True)

    # Lidar com valores não numéricos em 'y'
    data["y"] = pd.to_numeric(data["y"], errors='coerce')

    # Remover linhas com valores NaN após a conversão
    data.dropna(subset=['ds', 'y'], inplace=True)

    # Salvar dados processados
    data.to_csv(processed_data_path, index=False)
    print("\n")
    print(f"Dados processados salvos em: {processed_data_path}")
    print("\n")

# Caminhos relativos para os arquivos de dados
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'temperaturas.csv'))
processed_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Processed_data', 'processed_data.csv'))

# Pré-processar os dados
preprocess_data(data_path, processed_data_path)