import pandas as pd

# Não necessário para este problema de TS em especifico

# def engineer_features(data_path, feature_data_path):
    # Carregar dados
    # data = pd.read_csv(data_path, sep=",")

    # Realizar engenharia de features caso queira
    # Exemplo: criação de novas colunas, transformação de dados, etc.
    # data['lag_1'] = data['target'].shift(1)
    # data['lag_2'] = data['target'].shift(2)

    # Remover NaNs gerados pela criação de características
    # data.dropna(inplace=True)

    # Salvar dados com novas características
    # data.to_csv(feature_data_path, index=False)
    # print(f"Dados com novas características salvos em: {feature_data_path}")