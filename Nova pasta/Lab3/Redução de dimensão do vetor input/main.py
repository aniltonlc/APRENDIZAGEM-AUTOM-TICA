import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Atividade 1: Carregar dataset e Redução de Dimensão
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar PCA para 2 componentes (conforme Atividade 1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Representação gráfica da nuvem de pontos
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Dígitos (0-9)')
plt.title("Nuvem de Pontos: Digits Dataset (PCA Reduzido)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()