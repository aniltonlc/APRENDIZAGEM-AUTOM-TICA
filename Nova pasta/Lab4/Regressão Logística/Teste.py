import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar
modelo = pickle.load(open("modelo.pkl", "rb"))
pipe = pickle.load(open("pipe.pkl", "rb"))
X_test, y_test = pickle.load(open("teste.pkl", "rb"))

# 2. Transformar e Prever
X_test_nca = pipe.transform(X_test)
y_pred = modelo.predict(X_test_nca)

# 3. Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Gráfico e Legenda
cmap = plt.get_cmap('jet', 10)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_nca[:, 0], X_test_nca[:, 1], c=y_test, cmap=cmap, alpha=0.7, edgecolors='k')

legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"{i}",
                          markerfacecolor=cmap(i), markersize=10, markeredgecolor='k') for i in range(10)]

plt.legend(handles=legend_elements, title="Dígitos", loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("NCA - Conjunto de Teste")
plt.show()