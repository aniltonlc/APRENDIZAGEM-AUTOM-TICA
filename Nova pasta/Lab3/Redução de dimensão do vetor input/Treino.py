from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 1. Carga dos dados
digits = load_digits()
X = digits.data
y = digits.target

# 2. Redução de dimensão (Atividade 1) - Necessário para o treino simplificado
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Divisão em Treino e Teste (Atividade 2)
# Usamos 0.33 para garantir que 1/3 dos dados seja para teste
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.01, random_state=42
)

# 4. Treino do Classificador KNN
# Criamos o modelo com 15 vizinhos (k=15)
knn = KNeighborsClassifier(n_neighbors=15)

# O comando .fit é o que executa o "treino" propriamente dito
knn.fit(X_train, y_train)

# Verificação básica do treino
print("Treino concluído com sucesso!")
print(f"Número de amostras usadas para treino: {len(X_train)}")
print(f"Precisão obtida no conjunto de treino: {knn.score(X_train, y_train):.2%}")