from ucimlrepo import fetch_ucirepo
import pickle

# 1. Carregar
modelo = pickle.load(open("modelo.pkl", "rb"))
pipe = pickle.load(open("pipe.pkl", "rb"))

# 2. Dados
digits = fetch_ucirepo(id=80)
X, y = digits.data.features, digits.data.targets

# 3. Exemplo (índice 15)
novo = X.iloc[[12]]
novo_nca = pipe.transform(novo)

# 4. Resultado
print("Previsão:", modelo.predict(novo_nca)[0])
print("Real:", y.iloc[12].values[0])