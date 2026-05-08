import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# --- ATIVIDADE 1: Carga e Redução de Dimensão ---
digits = load_digits()
X = digits.data
y = digits.target

# Redução de dimensão para 2 componentes para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Representação gráfica da nuvem de pontos (Atividade 1)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, edgecolor='k')
plt.colorbar(scatter, label='Dígito (0-9)')
plt.title("Visualização do Dataset Digits com PCA (n=2)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()