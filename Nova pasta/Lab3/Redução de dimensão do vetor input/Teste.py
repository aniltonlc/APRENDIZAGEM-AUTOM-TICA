import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

# 1. CARREGAMENTO DOS DADOS REAIS
digits = load_digits()
X = digits.data  # Pixels originais
y = digits.target  # Etiquetas reais (0-9)

# 2. REDUÇÃO DE DIMENSÃO (PCA)
# Transformamos os dados reais de 64D para 2D para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. DIVISÃO PARA TESTE
# Separamos 1/3 dos dados reais para validar o modelo
_, X_test, _, y_test = train_test_split(
    X_pca, y, test_size=0.33, random_state=42, stratify=y
)

# 4. VISUALIZAÇÃO DOS VALORES REAIS
plt.figure(figsize=(11, 8))
cmap_discreto = plt.get_cmap('tab10', 10)

# Desenhamos apenas os pontos (cada ponto é um dado real)
# A cor 'c=y_test' garante que a cor do ponto é o seu valor real
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                    edgecolors='k', cmap=cmap_discreto, s=50, alpha=0.8)

# 5. LEGENDA CIRCULAR (Representando as classes reais)
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"Dígito {i}",
                          markerfacecolor=cmap_discreto(i), markersize=10,
                          markeredgecolor='k') for i in range(10)]

plt.legend(handles=legend_elements, title="Valores Reais", loc='center left',
           bbox_to_anchor=(1, 0.5), frameon=True)

plt.title("Visualização do Conjunto de Teste (Apenas Valores Reais)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True, linestyle='--', alpha=0.5) # Adiciona uma grelha para facilitar a leitura
plt.tight_layout()
plt.show()

# Verificação no terminal
print(f"Total de amostras reais exibidas no teste: {len(y_test)}")