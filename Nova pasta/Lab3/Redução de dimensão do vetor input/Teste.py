# Atividade 3: Adaptação do código de teste e visualização

import numpy as np
from matplotlib import pyplot as plt

from Treino import k_neighbors, X_test, y_test, knn, pca_train


def plot_classification_results(X_test, y_test, model, pca_model):
    X_test_pca = pca_model.transform(X_test)

    # Criar malha (mesh) para o fundo colorido [cite: 577]
    h = 0.5  # tamanho do passo na malha
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predição para cada ponto da malha [cite: 578]
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')

    # Plotar os pontos de teste reais
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                          edgecolor='k', s=20, cmap='tab10')
    plt.title(f"Classificação Digits (k={k_neighbors}) com PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


plot_classification_results(X_test, y_test, knn, pca_train)