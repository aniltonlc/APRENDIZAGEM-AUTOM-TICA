import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. CARREGAMENTO E PREPARAÇÃO (Igual aos outros ficheiros para consistência)
digits = load_digits()
X = digits.data
y = digits.target

# 2. REDUÇÃO DE DIMENSÃO (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. DIVISÃO TREINO/TESTE (90% Treino para maior precisão)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.1, random_state=42, stratify=y
)

# 4. TREINO DO MODELO KNN
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

# --- SISTEMA DE PREVISÃO CORRIGIDO (SEM RESHAPE) ---

print("=== SISTEMA DE PREVISÃO DE DÍGITOS ===")

# Selecionamos uma amostra do conjunto de teste pelo índice
indice_escolhido = 0
amostra_2d = [X_test[indice_escolhido]] # Envolvemos em [] para criar matriz 2D

# O modelo faz a previsão usando a lista de lista [[x, y]]
previsao = knn.predict(amostra_2d)

# Calculamos as probabilidades (confiança do modelo)
# Também usamos [] para evitar o erro de dimensão
probabilidades = knn.predict_proba(amostra_2d)
confianca = np.max(probabilidades) * 100

# 5. APRESENTAÇÃO DOS RESULTADOS
print(f"\nDados da Amostra (Índice {indice_escolhido}):")
print(f"-> Coordenadas (X, Y): {amostra_2d[0]}")
print(f"-> Valor REAL (Gabarito): {y_test[indice_escolhido]}")
print(f"-> Valor PREVISTO pelo KNN: {previsao[0]}")
print(f"-> Grau de Confiança: {confianca:.2f}%")

if previsao[0] == y_test[indice_escolhido]:
    print("\n✅ Resultado: O modelo acertou a previsão!")
else:
    print("\n❌ Resultado: O modelo falhou a previsão.")