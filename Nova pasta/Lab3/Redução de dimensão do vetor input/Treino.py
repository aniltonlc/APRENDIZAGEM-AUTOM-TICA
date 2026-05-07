# Atividade 2: Adaptação do código de treino
# Divisão dos dados (1/3 para teste)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from main import y, X

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Criar e treinar o modelo KNN (usando k=15 como no exemplo Iris) [cite: 552]
k_neighbors = 15
knn = KNeighborsClassifier(n_neighbors=k_neighbors)

# O modelo deve ser treinado com os dados transformados pelo PCA
pca_train = PCA(n_components=2)
X_train_pca = pca_train.fit_transform(X_train)
knn.fit(X_train_pca, y_train)

# Avaliação inicial
accuracy = knn.score(pca_train.transform(X_test), y_test)
print(f"Precisão do Modelo KNN com PCA: {accuracy:.2%}")