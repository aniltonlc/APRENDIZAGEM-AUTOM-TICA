from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle

# 1. Dados
digits = fetch_ucirepo(id=80)
X, y = digits.data.features, digits.data.targets.values.ravel()

# 2. Divisão (1/3 para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Pipeline e Treino (Atividade 1 e 2)
nca_pipe = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=42))
X_train_nca = nca_pipe.fit_transform(X_train, y_train)

modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train_nca, y_train)

# 4. Guardar
pickle.dump(modelo, open("modelo.pkl", "wb"))
pickle.dump(nca_pipe, open("pipe.pkl", "wb"))
pickle.dump((X_test, y_test), open("teste.pkl", "wb"))
print("Treino concluído.")