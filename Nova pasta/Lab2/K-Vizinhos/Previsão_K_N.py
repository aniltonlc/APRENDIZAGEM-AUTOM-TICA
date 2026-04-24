import pandas as pd
import pickle as p1
from sklearn import linear_model
import numpy as np

# Garantir que ucimlrepo esteja instalado e importado
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    print("ucimlrepo não encontrado, instalando...")

    from ucimlrepo import fetch_ucirepo

# Obter o conjunto de dados para obter os nomes das colunas para o DataFrame de entrada
digits_data_pred = fetch_ucirepo(id=80)
digits_data_X_full = digits_data_pred.data.features
digits_data_Y_full = digits_data_pred.data.targets # Para comparação potencial

# Carregar o modelo treinado
digits_model_loaded_pred = p1.load(open('digits_predictor', 'rb'))

# Obter os nomes das colunas para o DataFrame de entrada
digits_col = list(digits_data_X_full.columns)

# Orientar o usuário sobre o formato de entrada
print(f"Por favor, introduza {len(digits_col)} valores numéricos que representam as características de um dígito.")
print(f"Exemplo para o primeiro dígito de amostra (valor real: {digits_data_Y_full.iloc[0].values[0]}):")
print(f"{digits_data_X_full.iloc[0].values.tolist()}")

digits_input_values = list(map(float, input("introduza valores do dígito (separados por espaço):\n").split()))

# Garantir que a entrada corresponda ao número de características
if len(digits_input_values) != len(digits_col):
    print(f"Erro: Esperava {len(digits_col)} valores, mas obteve {len(digits_input_values)}.")
else:
    digits_input_df = pd.DataFrame([digits_input_values], columns=digits_col)

    print("\nCaracterísticas do dígito de entrada:")
    print(digits_input_df.iloc[0])

    digits_y_pred_single = digits_model_loaded_pred.predict(digits_input_df)

    predicted_digit = int(np.round(digits_y_pred_single[0, 0]))

    print(f"\nDígito previsto: {predicted_digit}")

    # Pode comparar manualmente com um dígito real conhecido se introduzir as suas características
    # Por exemplo, se introduzir as características de digits_data_X_full.iloc[0],
    # o dígito real seria digits_data_Y_full.iloc[0].values[0].