import pandas as pd
import pickle as p1
from sklearn import linear_model

# import numpy as np
dados_importar = "0;0;5;13;9;1;0;0;0;0;13;15;10;15;5;0;0;3;15;2;0;11;8;0;0;4;12;0;0;8;8;0;0;5;8;0;0;9;8;0;0;4;11;0;1;12;7;0;0;2;14;5;10;12;0;0;0;0;6;13;10;0;0;0"
dados_importar =dados_importar.replace(";", " ")

try:
    entrada = dados_importar
    data_x = list(map(float, entrada.split()))

    if len(data_x) == n_parametros:
        data_x_pd = pd.DataFrame([data_x])
        y_pred_manual = Modelogravado.predict(data_x_pd)
        print(f"\nPrevisão: {int(np.round(y_pred_manual[0]))}")
    else:
        print(f"Erro: Esperados {n_parametros} valores, recebidos {len(data_x)}")
except Exception as e2:
    print(f"Erro na entrada manual: {e2}")